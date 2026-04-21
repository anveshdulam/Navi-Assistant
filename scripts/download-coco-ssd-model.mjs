import fs from 'node:fs/promises';
import path from 'node:path';

const MODEL_URL =
  process.env.COCO_SSD_REMOTE_URL ??
  'https://storage.googleapis.com/tfjs-models/savedmodel/ssdlite_mobilenet_v2/model.json';

const OUTPUT_DIR = path.join(process.cwd(), 'public', 'models', 'coco-ssd');

const fetchText = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url} (${response.status})`);
  }
  return response.text();
};

const fetchBinary = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to download ${url} (${response.status})`);
  }
  const buffer = await response.arrayBuffer();
  return Buffer.from(buffer);
};

const ensureDir = async (dirPath) => {
  await fs.mkdir(dirPath, { recursive: true });
};

const main = async () => {
  await ensureDir(OUTPUT_DIR);

  const modelJsonText = await fetchText(MODEL_URL);
  const modelJsonPath = path.join(OUTPUT_DIR, 'model.json');
  await fs.writeFile(modelJsonPath, modelJsonText);

  let parsed;
  try {
    parsed = JSON.parse(modelJsonText);
  } catch {
    throw new Error('Downloaded model.json is not valid JSON');
  }

  const manifests = Array.isArray(parsed.weightsManifest) ? parsed.weightsManifest : [];
  const weightPaths = new Set();
  for (const group of manifests) {
    const paths = Array.isArray(group.paths) ? group.paths : [];
    for (const filePath of paths) {
      weightPaths.add(filePath);
    }
  }

  const baseUrl = new URL('.', MODEL_URL);
  for (const filePath of weightPaths) {
    const fileUrl = new URL(filePath, baseUrl).toString();
    const outputPath = path.join(OUTPUT_DIR, filePath);
    await ensureDir(path.dirname(outputPath));
    const data = await fetchBinary(fileUrl);
    await fs.writeFile(outputPath, data);
    console.log(`Downloaded ${filePath}`);
  }

  console.log('COCO-SSD model downloaded to public/models/coco-ssd');
};

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
