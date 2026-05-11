# Document Loading

Flowgentra can load documents from various sources and formats, automatically extracting text content and metadata for use in RAG systems.

## Supported Formats

Flowgentra supports loading documents from:

- **Plain text** (`.txt`)
- **Markdown** (`.md`) - preserves structure
- **HTML** (`.html`) - strips tags, preserves text
- **PDF** (`.pdf`) - extracts text content
- **JSON** (`.json`) - structured data
- **CSV** (`.csv`) - tabular data

## Loading Single Documents

=== "Python"

    ```python
    from flowgentra_ai.document_loaders import load_document

    # Load any supported format
    doc = load_document("./research_paper.pdf")
    print(f"Filename: {doc.filename}")
    print(f"Content length: {len(doc.content)}")
    print(f"File type: {doc.file_type}")

    # Access metadata
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Author: {doc.metadata.get('author')}")
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::data::load_document;
    use std::path::Path;

    let doc = load_document(Path::new("./research_paper.pdf"))?;
    println!("Loaded: {} ({} chars)", doc.filename, doc.content.len());
    ```

## Loading Directories

Load all documents from a directory recursively:

=== "Python"

    ```python
    from flowgentra_ai.document_loaders import load_directory

    # Load all documents from directory
    documents = load_directory("./docs")

    for doc in documents:
        print(f"Loaded: {doc.filename} ({doc.file_type})")
        print(f"Content preview: {doc.content[:100]}...")
    ```

=== "Rust"

    ```rust
    use flowgentra_ai::data::load_directory;

    let documents = load_directory("./docs")?;
    for doc in documents {
        println!("Loaded: {} ({})", doc.filename, doc.file_type);
    }
    ```

## Ingestion Pipeline

For production use, use the ingestion pipeline which provides progress tracking and error handling:

=== "Python"

    ```python
    from flowgentra_ai.document_loaders import IngestionPipeline

    # Create pipeline
    pipeline = IngestionPipeline()

    # Ingest documents with progress tracking
    documents = ["doc1.pdf", "doc2.md", "doc3.html"]
    stats = pipeline.ingest(documents)

    print(f"Processed: {stats.total_documents}")
    print(f"Successful: {stats.successful}")
    print(f"Failed: {stats.failed}")
    print(f"Total characters: {stats.total_characters}")
    ```

## PDF Processing

Special handling for PDF files:

=== "Python"

    ```python
    from flowgentra_ai.document_loaders import extract_pdf

    # Extract text from PDF
    text = extract_pdf("./document.pdf")
    print(f"Extracted {len(text)} characters")

    # PDF documents include page information in metadata
    doc = load_document("./document.pdf")
    print(f"Pages: {doc.metadata.get('pages')}")
    ```

## Custom Document Types

For unsupported formats, you can create custom loaders:

=== "Python"

    ```python
    from flowgentra_ai.document_loaders import LoadedDocument, FileType

    def load_custom_format(file_path: str) -> LoadedDocument:
        # Your custom loading logic here
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return LoadedDocument(
            filename=file_path,
            content=content,
            file_type=FileType.TEXT,  # or appropriate type
            metadata={"custom": True}
        )
    ```

## Error Handling

Document loading can fail for various reasons:

=== "Python"

    ```python
    from flowgentra_ai.document_loaders import load_document

    try:
        doc = load_document("./missing_file.pdf")
    except FileNotFoundError:
        print("File not found")
    except Exception as e:
        print(f"Loading failed: {e}")
    ```

## Best Practices

1. **Use ingestion pipeline** for batch processing with progress tracking
2. **Check file sizes** - very large files may need chunking before processing
3. **Handle encoding** - specify encoding for text files when needed
4. **Validate content** - check that extracted text is meaningful
5. **Use metadata** - leverage title, author, and other metadata for better retrieval</content>
<parameter name="filePath">c:\Users\OussamaBenHariz\Desktop\agentflow-rs\flowgentra-ai-py\docs\guides\document_loaders.md