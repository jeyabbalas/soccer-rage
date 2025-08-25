import fs from 'fs';
import path from 'path';
import Papa from 'papaparse';
import { pipeline } from '@huggingface/transformers';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const BI_ENCODER_MODEL = "onnx-community/Qwen3-Embedding-0.6B-ONNX";
const INPUT_CSV = path.join(__dirname, '..', 'public', 'data', 'onet_25-0_embedding.csv');
const OUTPUT_JSON = path.join(__dirname, '..', 'public', 'data', 'onet_25-0_vectordb.json');
const BATCH_SIZE = 16; // Adjust based on your machine's RAM

/**
 * Reads and parses a CSV file.
 * @param {string} filePath Path to the CSV file.
 * @returns {Promise<Array<Object>>} A promise that resolves to an array of objects.
 */
function parseCSV(filePath) {
    console.log(`Reading CSV from: ${filePath}`);
    const csvFile = fs.readFileSync(filePath, 'utf8');
    return new Promise(resolve => {
        Papa.parse(csvFile, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
                console.log(`Successfully parsed ${results.data.length} rows from CSV.`);
                resolve(results.data);
            }
        });
    });
}

/**
 * Main function to generate embeddings and build the vector database file.
 */
async function buildVectorDatabase() {
    console.log("🚀 Starting Step 2: Building Document Vector Database...");

    // Load and parse the source data
    const documents = await parseCSV(INPUT_CSV);
    const textsToEmbed = documents.map(doc => doc.embedding_text);

    // Load the feature extraction pipeline
    console.log(`Loading bi-encoder model: ${BI_ENCODER_MODEL}...`);
    const extractor = await pipeline(
        "feature-extraction",
        BI_ENCODER_MODEL,
        {
            quantized: false // full precision
        }
    );
    console.log("Model loaded successfully.");

    // Generate embeddings in batches to manage memory
    console.log(`Generating embeddings for ${textsToEmbed.length} documents in batches of ${BATCH_SIZE}...`);
    const vectorDB = [];
    const startTime = Date.now();

    for (let i = 0; i < textsToEmbed.length; i += BATCH_SIZE) {
        const batchTexts = textsToEmbed.slice(i, i + BATCH_SIZE);
        const batchDocs = documents.slice(i, i + BATCH_SIZE);

        const output = await extractor(batchTexts, {
            pooling: 'mean', // 'mean' pooling is a common choice for sentence embeddings
            normalize: true
        });

        const embeddings = output.tolist();

        for (let j = 0; j < batchDocs.length; j++) {
            vectorDB.push({
                soc: batchDocs[j]['O*NET-SOC Code'],
                title: batchDocs[j]['Title'],
                // embedding_text is large, so it is omitted from the final DB file
                // The web app can retrieve it from the original CSV if needed
                embedding: embeddings[j]
            });
        }

        const progress = Math.min(((i + BATCH_SIZE) / textsToEmbed.length) * 100, 100);
        console.log(`Processed batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(textsToEmbed.length / BATCH_SIZE)} (${progress.toFixed(2)}%)`);
    }

    const duration = (Date.now() - startTime) / 1000;
    console.log(`\nGenerated ${vectorDB.length} embeddings in ${duration.toFixed(2)} seconds.`);

    // 4. Save the final array to a JSON file
    console.log(`Saving vector database to: ${OUTPUT_JSON}`);
    fs.writeFileSync(OUTPUT_JSON, JSON.stringify(vectorDB));

    console.log("✅ Vector database built and saved successfully!");
}

// Run the main function
buildVectorDatabase().catch(error => {
    console.error("❌ An error occurred during vector database creation:", error);
});