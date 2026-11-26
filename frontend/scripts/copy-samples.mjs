// Copy the synthetic sample WAVs from data/samples into public/samples so the
// "Try a sample" button can fetch them. Runs before dev and build; the copies
// are git-ignored so the WAVs are only committed once.
import { copyFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const source = join(root, "..", "data", "samples");
const target = join(root, "public", "samples");
mkdirSync(target, { recursive: true });
const files = readdirSync(source).filter((f) => f.endsWith(".wav"));
for (const file of files) copyFileSync(join(source, file), join(target, file));
console.log(`Copied ${files.length} sample(s) to public/samples`);
