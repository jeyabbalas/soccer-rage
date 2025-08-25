import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);


// Main
(async () => {
    console.log("🚀 Starting O*NET data preparation script...");

    // Define paths relative to the current script's location
    const sourceDir = path.join(__dirname, '..', 'data', 'db_25_0_text');
    const outputDir = path.join(__dirname, '..', 'public', 'data');
    const outputFileName = 'onet_25-0_embedding.csv';
    const outputFilePath = path.join(outputDir, outputFileName);

    // Ensure the output directory exists before writing to it
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }

    // Run the main processing function
    await prepareOnetDataForEmbedding(sourceDir, outputFilePath);

    console.log("✅ Script finished successfully.");
})();


// Helper & Core Logic Functions

/**
 * Main function to read, process, and combine O*NET data.
 * @param {string} sourceDir The directory containing the source O*NET .txt files.
 * @param {string} outputFilePath The full path for the output CSV file.
 */
async function prepareOnetDataForEmbedding(sourceDir, outputFilePath) {
    console.log('Step 1: Validating required files...');

    const requiredFileNames = [
        "Occupation Data.txt", "Alternate Titles.txt", "Sample of Reported Titles.txt",
        "Task Statements.txt", "Emerging Tasks.txt", "Knowledge.txt", "Skills.txt",
        "Technology Skills.txt"
    ];

    // Check for missing files before starting
    const missingFiles = requiredFileNames.filter(name => !fs.existsSync(path.join(sourceDir, name)));
    if (missingFiles.length > 0) {
        console.error(`❌ Error: Missing required files in '${sourceDir}': ${missingFiles.join(', ')}`);
        process.exit(1); // Exit the script with an error code
    }

    try {
        // Load all relevant files into memory
        console.log('Step 2: Reading files...');
        const fileContents = requiredFileNames.map(name => {
            const filePath = path.join(sourceDir, name);
            const text = fs.readFileSync(filePath, 'utf8');
            return { name, text };
        });

        // Parse TSV content into structured data
        console.log('Step 3: Parsing data...');
        const data = new Map();
        for (const { name, text } of fileContents) {
            const onBadLines = name === "Emerging Tasks.txt" ? 'skip' : 'error';
            data.set(name, parseTsv(text, onBadLines));
        }

        // Aggregate data for each SOC Code
        console.log('Step 4: Aggregating data...');
        const allTitles = [
            ...data.get("Alternate Titles.txt"),
            ...data.get("Sample of Reported Titles.txt").map(r => ({ ...r, 'Alternate Title': r['Reported Job Title'] }))
        ];
        const aggTitles = aggregateBySoc(allTitles, 'Alternate Title');

        const taskData = data.get("Task Statements.txt").map(r => ({ ...r, Task: r.Task.replace(/\.$/, '') }));
        const emergingTaskData = data.get("Emerging Tasks.txt").map(r => ({ ...r, Task: r.Task.replace(/\.$/, '') }));
        const aggTasks = aggregateBySoc([...taskData, ...emergingTaskData], 'Task');

        const aggKnowledge = aggregateBySoc(data.get("Knowledge.txt"), 'Element Name');
        const aggSkills = aggregateBySoc(data.get("Skills.txt"), 'Element Name');

        const techSkillsData = data.get("Technology Skills.txt").map(r => ({
            ...r,
            Tech: `${r.Example} (${r['Commodity Title']})`
        }));
        const aggTech = aggregateBySoc(techSkillsData, 'Tech');

        // Merge all aggregated data into the base occupations data
        console.log('Step 5: Merging data sources...');
        let df = data.get("Occupation Data.txt");

        df.forEach(occupation => {
            const soc = occupation['O*NET-SOC Code'];
            occupation['All Titles'] = aggTitles.get(soc) || [];
            occupation['Tasks'] = aggTasks.get(soc) || [];
            occupation['Knowledge'] = aggKnowledge.get(soc) || [];
            occupation['Skills'] = aggSkills.get(soc) || [];
            occupation['Technologies'] = aggTech.get(soc) || [];
        });

        // Construct the final text document for embedding
        console.log('Step 6: Constructing final documents for embedding...');
        df.forEach(row => {
            row.embedding_text = createEmbeddingText(row);
        });

        // Prepare and save the final output
        const finalData = df.map(row => ({
            'O*NET-SOC Code': row['O*NET-SOC Code'],
            'Title': row['Title'],
            'embedding_text': row['embedding_text']
        }));

        const csvContent = convertToCsv(finalData);

        // Write the file directly to the output path
        console.log(`Step 7: Writing output to '${outputFilePath}'...`);
        fs.writeFileSync(outputFilePath, csvContent, 'utf8');

        console.log(`\n🎉 Successfully created file with ${finalData.length} professions.`);

        // Print a sample to the console for verification
        const sample = finalData.find(d => d['O*NET-SOC Code'] === '15-1111.00');
        if (sample) {
            console.log("\n--- Sample Document for '15-1111.00: Computer and Information Research Scientists' ---");
            console.log(sample.embedding_text);
            console.log("--------------------------------------------------------------------------------------");
        }

    } catch (error) {
        console.error("❌ An error occurred during processing:", error);
        process.exit(1);
    }
}

// The following functions are pure logic and require no changes
function parseTsv(text, onBadLine = 'skip') {
    const lines = text.trim().split('\n');
    if (lines.length < 2) return [];

    const headers = lines[0].trim().split('\t').map(h => h.trim());
    const socCodeHeader = headers[0];
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].trim().split('\t');

        if (values.length !== headers.length) {
            if (onBadLine === 'skip') {
                console.warn(`Skipping malformed line ${i + 1}: Expected ${headers.length} columns, but found ${values.length}.`);
                continue;
            }
        }

        const row = {};
        headers.forEach((header, index) => {
            const key = (header === socCodeHeader) ? 'O*NET-SOC Code' : header;
            row[key] = values[index] ? values[index].trim() : '';
        });
        data.push(row);
    }
    return data;
}

function aggregateBySoc(data, valueKey) {
    const groups = new Map();
    for (const item of data) {
        const soc = item['O*NET-SOC Code'];
        const value = item[valueKey];
        if (!soc || !value) continue;
        if (!groups.has(soc)) {
            groups.set(soc, new Set());
        }
        groups.get(soc).add(value);
    }
    const finalGroups = new Map();
    for (const [soc, valueSet] of groups.entries()) {
        finalGroups.set(soc, Array.from(valueSet));
    }
    return finalGroups;
}

function convertToCsv(data) {
    if (!data || data.length === 0) return "";
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];

    const escapeField = (field) => {
        const str = String(field);
        if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replace(/"/g, '""')}"`;
        }
        return str;
    };

    data.forEach(row => {
        const values = headers.map(header => escapeField(row[header]));
        csvRows.push(values.join(','));
    });
    return csvRows.join('\n');
}

function createEmbeddingText(row) {
    let text = `Official Title: ${row['Title']}\n`;
    text += `Description: ${row['Description']}\n\n`;
    if (row['All Titles']?.length > 0) {
        text += `Alternate and Reported Titles: ${row['All Titles'].join(', ')}\n\n`;
    }
    if (row['Tasks']?.length > 0) {
        text += "Key Tasks:\n" + row['Tasks'].slice(0, 10).map(task => `- ${task}\n`).join('');
    }
    if (row['Skills']?.length > 0) {
        text += `\nKey Skills: ${row['Skills'].join(', ')}\n`;
    }
    if (row['Knowledge']?.length > 0) {
        text += `Required Knowledge: ${row['Knowledge'].join(', ')}\n`;
    }
    if (row['Technologies']?.length > 0) {
        text += `\nTechnologies and Software Used: ${row['Technologies'].join(', ')}\n`;
    }
    return text.trim();
}