// DOCS の埋め込みベクトルを事前計算し、parquet として public/embeddings/ に出力する。
// GitHub Actions(または手元)で実行し、生成物を GitHub Pages から配信する。
// ブラウザ側はこの parquet を読み込むことで、一番重い「DOCS の埋め込み計算」をスキップできる。
//
// 使い方: npm run precompute
//
// 現状は Transformers.js 系モデル(PRECOMPUTE_MODEL_IDS)のみ対象。
// MediaPipe 系はブラウザ前提(WASM)で Node での再現が困難なため対象外。
import { mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { DuckDBInstance } from "@duckdb/node-api";

import { DOCS, parquetFileNameFor } from "../src/lib/docs.ts";
import { EMBEDDING_MODELS, PRECOMPUTE_MODEL_IDS } from "../src/lib/models.ts";
import { createTransformersEmbedder } from "../src/lib/embedder.ts";

const OUTPUT_DIR = join(process.cwd(), "public", "embeddings");

// 埋め込み済みベクトル + 本文を 1 つの parquet に書き出す。
// id INTEGER, content VARCHAR, vec FLOAT[dim] のスキーマ。
const writeParquet = async (
  outPath: string,
  rows: { id: number; content: string; vec: number[] }[],
  dim: number
): Promise<void> => {
  // 一時 JSON に書き出し、DuckDB の read_json 経由で parquet 化する。
  // こうすると SQL 文字列のエスケープ(本文中の ' など)を自前で扱わずに済む。
  const tmpJson = join(tmpdir(), `precompute-${Date.now()}.json`);
  await writeFile(tmpJson, JSON.stringify(rows), "utf8");

  const instance = await DuckDBInstance.create();
  const conn = await instance.connect();
  try {
    await conn.run(
      `COPY (
         SELECT id, content, vec::FLOAT[${dim}] AS vec
         FROM read_json('${tmpJson}', format='array')
         ORDER BY id
       ) TO '${outPath}' (FORMAT parquet);`
    );
  } finally {
    conn.closeSync();
    instance.closeSync();
    await rm(tmpJson, { force: true });
  }
};

const precomputeModel = async (modelId: string): Promise<void> => {
  const model = EMBEDDING_MODELS.find((m) => m.id === modelId);
  if (!model) {
    console.warn(`[skip] 未知のモデル id: ${modelId}`);
    return;
  }
  if (model.backend !== "transformers") {
    console.warn(
      `[skip] ${modelId}: backend=${model.backend} は Node での事前計算に未対応`
    );
    return;
  }

  console.log(`\n=== ${modelId} (${model.modelId}) ===`);
  console.time(`embed ${modelId}`);
  const embedder = await createTransformersEmbedder(
    model.modelId,
    model.dtype,
    model.pooling,
    model.usePrefix
  );

  const rows: { id: number; content: string; vec: number[] }[] = [];
  for (let i = 0; i < DOCS.length; i++) {
    const vec = await embedder.embed(DOCS[i], "passage");
    rows.push({ id: i, content: DOCS[i], vec });
  }
  embedder.close();
  console.timeEnd(`embed ${modelId}`);

  const dim = rows[0]?.vec.length ?? embedder.dim;
  if (!dim) throw new Error(`${modelId}: 埋め込み次元の判定に失敗しました`);

  const outPath = join(OUTPUT_DIR, parquetFileNameFor(modelId));
  await writeParquet(outPath, rows, dim);
  console.log(`-> ${outPath} (dim=${dim}, rows=${rows.length})`);
};

const main = async () => {
  await mkdir(OUTPUT_DIR, { recursive: true });
  for (const modelId of PRECOMPUTE_MODEL_IDS) {
    await precomputeModel(modelId);
  }
  console.log("\n完了: 事前計算した parquet を public/embeddings/ に出力しました");
};

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
