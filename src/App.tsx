import { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

import * as duckdb from "@duckdb/duckdb-wasm";
import duckdb_worker from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?worker";
import duckdb_wasm from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import type { Table } from "apache-arrow";

import { DOCS, cacheKeyFor, parquetFileNameFor } from "./lib/docs";
import {
  EMBEDDING_MODELS,
  DEFAULT_MODEL_ID,
  type ModelConfig,
} from "./lib/models";
import { type Embedder, createTransformersEmbedder } from "./lib/embedder";

// MediaPipe の wasm は CDN から取得する。node_modules 相対パスはビルド後(本番)では解決できないため。
const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@0.10.35/wasm";

// 事前計算済み(precompute)埋め込み parquet の URL。GitHub Pages 上の base 配下に配置される。
// 例: /study-duckdb-wasm-vss/embeddings/granite-97m-r2__<hash>.parquet
const parquetUrlFor = (modelId: string) =>
  `${import.meta.env.BASE_URL}embeddings/${parquetFileNameFor(modelId)}`;

// 検索結果 1 行。sora_doc の id/content に距離を加えたもの。
type ResultRow = { id: number; content: string; distance: number };

// OPFS に DOCS の埋め込みベクトルをキャッシュする。読めなければ null を返し、書き込み失敗は握りつぶす
// (OPFS 非対応環境でも動くように)。これにより再読込・モデル再選択時の再埋め込みをスキップできる。
const OPFS_CACHE_DIR = "vss-embeddings-cache";

const readEmbeddingsCache = async (
  key: string
): Promise<number[][] | null> => {
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(OPFS_CACHE_DIR, { create: true });
    const fh = await dir.getFileHandle(`${key}.json`);
    const file = await fh.getFile();
    return JSON.parse(await file.text());
  } catch {
    return null;
  }
};

const writeEmbeddingsCache = async (
  key: string,
  vectors: number[][]
): Promise<void> => {
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle(OPFS_CACHE_DIR, { create: true });
    const fh = await dir.getFileHandle(`${key}.json`, { create: true });
    const writable = await fh.createWritable();
    await writable.write(JSON.stringify(vectors));
    await writable.close();
  } catch (e) {
    console.warn("OPFS への埋め込みキャッシュ書き込みに失敗:", e);
  }
};

// MediaPipe バックエンドの Embedder を生成する。
const createMediaPipeEmbedder = async (url: string): Promise<Embedder> => {
  const { TextEmbedder, FilesetResolver } = await import(
    "@mediapipe/tasks-text"
  );
  const textFiles = await FilesetResolver.forTextTasks(MEDIAPIPE_WASM);
  const te = await TextEmbedder.createFromOptions(textFiles, {
    baseOptions: { modelAssetPath: url },
  });
  const probe = te.embed("dimension probe");
  const dim = probe.embeddings[0].floatEmbedding?.length ?? 0;
  return {
    dim,
    embed: async (text) =>
      Array.from(te.embed(text).embeddings[0].floatEmbedding ?? []),
    close: () => te.close(),
  };
};

const createEmbedder = (model: ModelConfig): Promise<Embedder> => {
  if (model.backend === "mediapipe") {
    return createMediaPipeEmbedder(model.url);
  }
  return createTransformersEmbedder(
    model.modelId,
    model.dtype,
    model.pooling,
    model.usePrefix
  );
};

// GitHub Pages に置かれた事前計算済み parquet から DOCS の埋め込みを読み込む。
// 見つからない/読めない場合は null を返し、呼び出し側でその場計算にフォールバックする。
const loadParquetVectors = async (
  db: duckdb.AsyncDuckDB,
  modelId: string
): Promise<number[][] | null> => {
  const url = parquetUrlFor(modelId);
  // まず HEAD で存在確認(404 のときに parquet パースを試みて警告を出すのを避ける)
  try {
    const head = await fetch(url, { method: "HEAD" });
    if (!head.ok) return null;
  } catch {
    return null;
  }

  const fileName = parquetFileNameFor(modelId);
  try {
    // モデル再選択で再登録されても問題ないよう、先に dropFile しておく
    await db.dropFile(fileName).catch(() => {});
    await db.registerFileURL(
      fileName,
      url,
      duckdb.DuckDBDataProtocol.HTTP,
      false
    );
    const conn = await db.connect();
    try {
      const tbl = await conn.query(
        `SELECT vec FROM read_parquet('${fileName}') ORDER BY id`
      );
      const vecCol = tbl.getChild("vec");
      if (!vecCol || tbl.numRows !== DOCS.length) return null;
      const vectors: number[][] = [];
      for (let i = 0; i < tbl.numRows; i++) {
        const cell = vecCol.get(i) as Iterable<number> | null;
        if (!cell) return null;
        vectors.push(Array.from(cell, Number));
      }
      return vectors;
    } finally {
      await conn.close();
    }
  } catch (e) {
    console.warn("parquet の読み込みに失敗。埋め込みを計算します:", e);
    return null;
  }
};

// 例文を埋め込み、DuckDB に VSS 用テーブルと HNSW インデックスを構築する。
// DOCS の埋め込み(一番重い処理)は次の優先順位で解決し、重い計算を可能な限り避ける:
//   1. OPFS キャッシュ (端末ローカル・最速)
//   2. 事前計算済み parquet (GitHub Pages・CI で計算済み)
//   3. その場で埋め込み計算 (フォールバック)
const buildDatabase = async (
  embedder: Embedder,
  modelId: string
): Promise<{
  db: duckdb.AsyncDuckDB;
  dim: number;
  source: "cache" | "precomputed" | "computed";
}> => {
  const worker = new duckdb_worker();
  const logger = new duckdb.VoidLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(duckdb_wasm);
  await db.open({});

  const conn = await db.connect();
  await conn.query("INSTALL vss;");
  await conn.query("LOAD vss;");

  const cacheKey = cacheKeyFor(modelId);
  let docVectors = await readEmbeddingsCache(cacheKey);
  let source: "cache" | "precomputed" | "computed";
  if (docVectors && docVectors.length === DOCS.length) {
    source = "cache";
    console.log("DOCS の埋め込みを OPFS キャッシュから復元");
  } else {
    docVectors = await loadParquetVectors(db, modelId);
    if (docVectors) {
      await writeEmbeddingsCache(cacheKey, docVectors);
      source = "precomputed";
      console.log("DOCS の埋め込みを事前計算 parquet から復元");
    } else {
      docVectors = [];
      for (const doc of DOCS) {
        docVectors.push(await embedder.embed(doc, "passage"));
      }
      await writeEmbeddingsCache(cacheKey, docVectors);
      source = "computed";
      console.log("DOCS の埋め込みを計算して OPFS に保存");
    }
  }

  const dim = docVectors[0]?.length ?? embedder.dim;
  if (!dim) throw new Error("埋め込み次元の判定に失敗しました");

  await conn.query("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;");
  await conn.query(
    "CREATE TABLE sora_doc (id INTEGER DEFAULT nextval('id_sequence') PRIMARY KEY, content VARCHAR);"
  );
  await conn.query(`CREATE TABLE embeddings (vec FLOAT[${dim}]);`);

  const placeholders = Array(dim).fill("?").join(",");
  const stmt2 = await conn.prepare(
    `INSERT INTO embeddings VALUES (ARRAY[${placeholders}]);`
  );

  for (let i = 0; i < DOCS.length; i++) {
    const stmt = await conn.prepare("INSERT INTO sora_doc (content) VALUES (?);");
    await stmt.query(DOCS[i]);
    await stmt.close();
    const vec = docVectors[i];
    if (vec && vec.length) {
      await stmt2.query(...vec);
    }
  }
  await stmt2.close();

  await conn.query(`CREATE INDEX hnsw_index ON embeddings USING HNSW (vec);`);
  await conn.close();

  return { db, dim, source };
};

function App() {
  const [query, setQuery] = useState("こんにちは！ベクトル検索のデモです");
  const [selectedModelId, setSelectedModelId] =
    useState<string>(DEFAULT_MODEL_ID);
  const [queryEmbedding, setQueryEmbedding] = useState<number[] | null>(null);
  const [myDuckDB, setMyDuckDB] = useState<duckdb.AsyncDuckDB | null>(null);
  const [embeddingDim, setEmbeddingDim] = useState<number | null>(null);
  const [embeddingSource, setEmbeddingSource] = useState<
    "cache" | "precomputed" | "computed" | null
  >(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading"
  );
  const [resultRows, setResultRows] = useState<ResultRow[]>([]);
  const debounceTimerRef = useRef<number | null>(null);
  // 旧インスタンスのクリーンアップ用
  const embedderRef = useRef<Embedder | null>(null);
  const dbRef = useRef<duckdb.AsyncDuckDB | null>(null);

  // モデル選択が変わるたびに、埋め込みモデルと DuckDB を作り直す。
  useEffect(() => {
    let cancelled = false;
    const rebuild = async () => {
      setStatus("loading");
      setResultRows([]);
      setQueryEmbedding(null);
      setMyDuckDB(null);
      setEmbeddingDim(null);
      setEmbeddingSource(null);

      // 旧インスタンスを破棄
      embedderRef.current?.close();
      embedderRef.current = null;
      if (dbRef.current) {
        await dbRef.current.terminate();
        dbRef.current = null;
      }

      try {
        const model =
          EMBEDDING_MODELS.find((m) => m.id === selectedModelId) ??
          EMBEDDING_MODELS[0];
        const embedder = await createEmbedder(model);
        if (cancelled) {
          embedder.close();
          return;
        }
        embedderRef.current = embedder;

        const { db, dim, source } = await buildDatabase(
          embedder,
          selectedModelId
        );
        if (cancelled) {
          await db.terminate();
          return;
        }
        dbRef.current = db;

        setMyDuckDB(db);
        setEmbeddingDim(dim);
        setEmbeddingSource(source);
        setStatus("ready");
        console.log(`Model "${selectedModelId}" ready (dim=${dim}, ${source})`);
      } catch (e) {
        if (!cancelled) {
          console.error("Failed to initialize embedding model:", e);
          setStatus("error");
        }
      }
    };
    void rebuild();
    return () => {
      cancelled = true;
    };
  }, [selectedModelId]);

  // クエリを debounce して埋め込む
  const debouncedEmbedQuery = useCallback((query: string) => {
    if (!embedderRef.current) return;

    if (debounceTimerRef.current !== null) {
      window.clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = window.setTimeout(async () => {
      const embedder = embedderRef.current;
      if (!embedder) return;
      const vec = await embedder.embed(query, "query");
      setQueryEmbedding(vec);
      debounceTimerRef.current = null;
    }, 800);
  }, []);

  useEffect(() => {
    if (!query) return;
    if (status !== "ready") return;

    debouncedEmbedQuery(query);

    return () => {
      if (debounceTimerRef.current !== null) {
        window.clearTimeout(debounceTimerRef.current);
      }
    };
  }, [query, status, debouncedEmbedQuery]);

  useEffect(() => {
    const doit = async () => {
      if (!myDuckDB) return;
      if (!queryEmbedding) return;
      if (!embeddingDim) return;

      const conn = await myDuckDB.connect();

      const sql = `
      SELECT c.*, array_distance(e.vec, [${queryEmbedding.join(
        ","
      )}]::FLOAT[${embeddingDim}]) AS distance
      FROM embeddings e
      JOIN sora_doc c ON e.rowid = c.rowid
      ORDER BY distance ASC
      LIMIT 10;
    `;
      const newResults: Table = await conn.query(sql);
      const newResultRows: ResultRow[] = newResults
        .toArray()
        .map((row) => JSON.parse(String(row)) as ResultRow);

      setResultRows(newResultRows);
      await conn.close();
    };
    void doit();
  }, [myDuckDB, queryEmbedding, embeddingDim]);

  return (
    <div>
      <h1>DuckDB WASM VSS with MediaPipe / Transformers.js - Demo App</h1>
      <p>
        This app demonstrates the integration of DuckDB VSS with text embeddings
        from MediaPipe and Transformers.js (multilingual models) in React.
      </p>
      <div>
        <h2 style={{ textAlign: "left" }}>Embedding model:</h2>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <select
            value={selectedModelId}
            onChange={(e) => setSelectedModelId(e.target.value)}
            disabled={status === "loading"}
            style={{ padding: "4px" }}
          >
            {EMBEDDING_MODELS.map((m) => (
              <option key={m.id} value={m.id}>
                {m.label}
              </option>
            ))}
          </select>
          {status === "ready" && embeddingDim ? (
            <span>
              次元数: {embeddingDim}
              {embeddingSource === "cache"
                ? "（OPFSキャッシュから復元）"
                : embeddingSource === "precomputed"
                ? "（事前計算 parquet から復元）"
                : embeddingSource === "computed"
                ? "（埋め込みを計算しOPFSに保存）"
                : ""}
            </span>
          ) : null}
          {status === "loading" ? (
            <span>モデル読み込み中...（大きいモデルは時間がかかります）</span>
          ) : null}
          {status === "error" ? (
            <span style={{ color: "red" }}>モデルの読み込みに失敗しました</span>
          ) : null}
        </div>
      </div>
      <div>
        <h2 style={{ textAlign: "left" }}>Query:</h2>
        <div style={{ display: "flex", alignItems: "left" }}>
          <input
            style={{ width: "300px" }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type your query here"
          />
        </div>
      </div>
      <>
        {status !== "ready" ? (
          <div>
            <h2 style={{ textAlign: "left" }}>
              {status === "loading"
                ? "Initializing... モデルとデータベースを準備しています"
                : "初期化に失敗しました"}
            </h2>
          </div>
        ) : null}
      </>
      <div>
        <h2 style={{ textAlign: "left" }}>Results:</h2>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            width: "100%",
            border: "1px solid #ddd",
            borderRadius: "4px",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              display: "flex",
              background: "#f5f5f5",
              fontWeight: "bold",
              padding: "10px 0",
              borderBottom: "1px solid #ddd",
            }}
          >
            <div style={{ flex: "0 0 50px", padding: "0 10px" }}>ID</div>
            <div style={{ flex: "0 0 100px", padding: "0 10px" }}>Distance</div>
            <div style={{ flex: "1", padding: "0 10px" }}>Content</div>
          </div>
          {resultRows.map((row, index) => (
            <div
              key={index}
              style={{
                display: "flex",
                padding: "10px 0",
                borderBottom:
                  index < resultRows.length - 1 ? "1px solid #eee" : "none",
              }}
            >
              <div style={{ flex: "0 0 50px", padding: "0 10px" }}>
                {row.id}
              </div>
              <div style={{ flex: "0 0 100px", padding: "0 10px" }}>
                {row.distance?.toFixed(4)}
              </div>
              <div style={{ flex: "1", padding: "0 10px", textAlign: "left" }}>
                {row.content}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
