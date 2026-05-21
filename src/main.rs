use anyhow::Result;
use clap::{Parser, Subcommand};
use ignore::WalkBuilder;
use rig::completion::Prompt;
use rig::embeddings::{Embed, EmbedError, EmbeddingModel, TextEmbedder};
use rig::OneOrMany;
use rig::providers::gemini::Client;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

/// Current Gemini embedding model (replaces deprecated text-embedding-004).
const GEMINI_EMBEDDING_MODEL: &str = "gemini-embedding-001";
/// Output dimensions for gemini-embedding-001 (supports MRL; 768 is a good RAG default).
const GEMINI_EMBEDDING_DIMS: usize = 768;
/// Chat model (1.5 models are retired on the Gemini API).
const GEMINI_CHAT_MODEL: &str = "gemini-2.0-flash";

#[derive(Parser)]
#[command(name = "rig-rag")]
#[command(about = "A RAG-powered AI agent for researching codebases using Rig", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest a codebase and start an interactive research session
    Chat {
        /// Path to the codebase directory
        path: PathBuf,
        /// Gemini model for chat (default: gemini-2.0-flash)
        #[arg(short, long, default_value = GEMINI_CHAT_MODEL)]
        model: String,
    },
}

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
struct CodeSnippet {
    path: String,
    content: String,
}

impl Embed for CodeSnippet {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.content.clone());
        Ok(())
    }
}

impl std::fmt::Display for CodeSnippet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "File: {}\n---\n{}\n---", self.path, self.content)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    let cli = Cli::parse();

    let client = Client::from_env();

    match cli.command {
        Commands::Chat { path, model } => {
            run_chat_session(&client, path, model).await?;
        }
    }

    Ok(())
}

async fn run_chat_session(client: &Client, path: PathBuf, model_name: String) -> Result<()> {
    println!("🚀 Ingesting codebase from: {:?}...", path);

    let mut snippets = Vec::new();
    let walker = WalkBuilder::new(&path)
        .hidden(true)
        .git_ignore(true)
        .build();

    for result in walker {
        let entry = match result {
            Ok(entry) => entry,
            Err(e) => {
                eprintln!("Error: {}", e);
                continue;
            }
        };

        if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            let file_path = entry.path();

            let ext = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
            let relevant_exts = [
                "rs", "ts", "js", "py", "go", "c", "cpp", "h", "md", "toml", "json", "yaml",
            ];

            if relevant_exts.contains(&ext) {
                if let Ok(content) = fs::read_to_string(file_path) {
                    if !content.is_empty() && content.len() < 20000 {
                        snippets.push(CodeSnippet {
                            path: file_path
                                .strip_prefix(&path)
                                .unwrap_or(file_path)
                                .to_string_lossy()
                                .to_string(),
                            content,
                        });
                    }
                }
            }
        }
    }

    if snippets.is_empty() {
        println!("❌ No relevant files found in the directory.");
        return Ok(());
    }

    println!(
        "📄 Found {} files. Generating embeddings using {}...",
        snippets.len(),
        GEMINI_EMBEDDING_MODEL
    );

    let embedding_model =
        client.embedding_model_with_ndims(GEMINI_EMBEDDING_MODEL, GEMINI_EMBEDDING_DIMS);

    // Gemini embedContent returns one vector per request; embed files individually.
    let total = snippets.len();
    let mut embeddings = Vec::with_capacity(total);
    for (i, snippet) in snippets.into_iter().enumerate() {
        print!("\r  Embedding {}/{}: {}...", i + 1, total, snippet.path);
        io::stdout().flush()?;
        let emb = embedding_model.embed_text(&snippet.content).await?;
        embeddings.push((snippet, OneOrMany::one(emb)));
    }
    println!();

    let vector_store =
        InMemoryVectorStore::from_documents_with_id_f(embeddings, |doc| doc.path.clone());

    println!("✅ Codebase indexed. Starting agent...");

    let index = vector_store.index(embedding_model);

    let agent = client
        .agent(&model_name)
        .preamble(
            "You are an expert Senior Software Engineer and Architect. 
        You have access to a codebase via a RAG system. 
        When answering questions, use the retrieved code snippets to provide accurate, specific, and technical advice.
        If you are unsure or the code doesn't contain the answer, say so.
        Always mention the file paths you are referring to.",
        )
        .dynamic_context(2, index)
        .build();

    println!("\n--- Codebase Research Agent ---");
    println!("Type 'exit' to quit.");

    loop {
        print!("\n> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let query = input.trim();

        if query.is_empty() {
            continue;
        }

        if query == "exit" || query == "quit" {
            break;
        }

        print!("🤔 Thinking...");
        io::stdout().flush()?;

        let response = agent.prompt(query).await?;

        println!("\r{}", response);
    }

    println!("Bye!");
    Ok(())
}
