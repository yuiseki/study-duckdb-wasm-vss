import { useEffect, useRef, useState, useCallback } from "react";
import "./App.css";

import * as duckdb from "@duckdb/duckdb-wasm";
import duckdb_worker from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?worker";
import duckdb_wasm from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import type { Table, StructRowProxy } from "apache-arrow";

// MediaPipe の wasm は CDN から取得する。node_modules 相対パスはビルド後(本番)では解決できないため。
const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-text@0.10.35/wasm";

// 埋め込みモデルの統一インターフェース。バックエンド(MediaPipe / Transformers.js)の差を吸収する。
// e5 系は "query: " / "passage: " の接頭辞が必要なため kind を受け取る。
type Embedder = {
  dim: number;
  embed: (text: string, kind: "query" | "passage") => Promise<number[]>;
  close: () => void;
};

type ModelConfig =
  | {
      id: string;
      label: string;
      backend: "mediapipe";
      url: string;
    }
  | {
      id: string;
      label: string;
      backend: "transformers";
      modelId: string;
      dtype: string;
      // プーリング方式。e5 系は mean、granite 系は cls
      pooling: "mean" | "cls";
      // e5 系は "query: " / "passage: " 接頭辞が必要。granite 系は不要
      usePrefix: boolean;
    };

// 注意: MediaPipe の Average Word / Universal Sentence Encoder は語彙ベースの英語専用で、
// 空白で区切られない日本語文はすべて未知語となり同一ベクトルに潰れる(=検索が機能しない)。
// 日本語で意味のある結果が得られるのは BERT(文字レベル分割) か Transformers.js の多言語モデル。
const EMBEDDING_MODELS: ModelConfig[] = [
  // granite 系(IBM, Apache-2.0)。多言語(日本語含む)対応、CLS プーリング、接頭辞なし。
  // ONNX は yuiseki 自身が Apache-2.0 明示で再ホストしたもの。
  {
    id: "granite-97m-r2",
    label: "granite-embedding-97m-multilingual-r2（多言語・Apache-2.0・約93MB・推奨）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-97m-multilingual-r2-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "granite-107m",
    label: "granite-embedding-107m-multilingual（多言語・Apache-2.0・約102MB）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-107m-multilingual-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "granite-278m",
    label: "granite-embedding-278m-multilingual（多言語・Apache-2.0・約265MB）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-278m-multilingual-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "granite-311m-r2",
    label: "granite-embedding-311m-multilingual-r2（多言語・Apache-2.0・約298MB・高精度）",
    backend: "transformers",
    modelId: "yuiseki/granite-embedding-311m-multilingual-r2-ONNX",
    dtype: "q8",
    pooling: "cls",
    usePrefix: false,
  },
  {
    id: "multilingual-e5-base",
    label: "multilingual-e5-base（多言語・MIT・約265MB）",
    backend: "transformers",
    modelId: "onnx-community/multilingual-e5-base-ONNX",
    dtype: "q8",
    pooling: "mean",
    usePrefix: true,
  },
  {
    id: "bert_embedder",
    label: "MediaPipe BERT（約26MB・日本語可・速い）",
    backend: "mediapipe",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite",
  },
  {
    id: "universal_sentence_encoder",
    label: "MediaPipe Universal Sentence Encoder（約6MB・英語のみ）",
    backend: "mediapipe",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite",
  },
  {
    id: "average_word_embedder",
    label: "MediaPipe Average Word Embedding（約0.7MB・最速・英語のみ）",
    backend: "mediapipe",
    url: "https://storage.googleapis.com/mediapipe-models/text_embedder/average_word_embedder/float32/1/average_word_embedder.tflite",
  },
];

const DEFAULT_MODEL_ID = "granite-97m-r2";

// ベクトル検索の対象となる例文。日本語・英語をいろいろなトピックで混在させている。
const DOCS = [
  "こんにちは！ベクトル検索のデモです",
  "This is a demo of vector search",
  "今日はとても良い天気です。",
  "Today is a very nice day.",
  "こんにちは、世界！",
  "Hello, world!",
  "ベクトル検索って何ですか？",
  "What is vector search?",
  "ベクトル検索は、情報検索の一種で、データをベクトル空間にマッピングし、類似性を測定する手法です。",
  "明日は雨が降るみたいなので、傘を持って出かけましょう。",
  "It looks like it will rain tomorrow, so let's bring an umbrella.",
  "週末は家族と一緒に近くの公園でピクニックをしました。",
  "We had a picnic with our family at a nearby park over the weekend.",
  "猫はソファの上で気持ちよさそうに昼寝をしている。",
  "The cat is taking a comfortable nap on the sofa.",
  "新しいスマートフォンはカメラの性能が大幅に向上した。",
  "The new smartphone has a significantly improved camera.",
  "京都には歴史的な寺院や神社がたくさんあります。",
  "Kyoto has many historical temples and shrines.",
  "機械学習モデルの学習には大量のデータと計算資源が必要だ。",
  "Training a machine learning model requires a lot of data and compute.",
  "朝食にトーストとコーヒーを楽しむのが日課です。",
  "Enjoying toast and coffee for breakfast is my daily routine.",
  "電車が遅延したため、会議に少し遅刻してしまった。",
  "The train was delayed, so I was a little late for the meeting.",
  "この本はプログラミング初心者にとてもおすすめです。",
  "This book is highly recommended for programming beginners.",
  "海辺で夕日を眺めるのはとてもロマンチックだ。",
  "Watching the sunset by the sea is very romantic.",
  "健康のために毎朝30分のジョギングを続けている。",
  "I keep jogging for 30 minutes every morning to stay healthy.",
  "データベースのインデックスは検索を高速化するために使われる。",
  "Database indexes are used to speed up searches.",
];

// DOCS の内容から安定したハッシュを作る。例文を変更するとキャッシュが自動的に無効化される。
const DOCS_HASH = (() => {
  let h = 0;
  const s = DOCS.join("");
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(36);
})();

const cacheKeyFor = (modelId: string) => `${modelId}__${DOCS_HASH}`;

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

// Transformers.js バックエンドの Embedder を生成する。
const createTransformersEmbedder = async (
  modelId: string,
  dtype: string,
  pooling: "mean" | "cls",
  usePrefix: boolean
): Promise<Embedder> => {
  const { pipeline } = await import("@huggingface/transformers");
  // dtype を指定して量子化版 ONNX をロードする(例: q8 -> model_quantized.onnx)
  const extractor = await pipeline("feature-extraction", modelId, {
    dtype: dtype as any,
  });
  const prefix = (kind: "query" | "passage") =>
    usePrefix ? (kind === "query" ? "query: " : "passage: ") : "";
  const run = async (text: string, kind: "query" | "passage") => {
    // モデルの定義に合わせた pooling + 正規化で 1 文 1 ベクトルにする
    // (e5 系=mean, granite 系=cls)
    const out = await extractor(prefix(kind) + text, {
      pooling,
      normalize: true,
    });
    return Array.from(out.data as Float32Array).map((v) => Number(v));
  };
  const probe = await run("probe", "query");
  return {
    dim: probe.length,
    embed: run,
    close: () => {
      void (extractor as any).dispose?.();
    },
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

// 例文を埋め込み、DuckDB に VSS 用テーブルと HNSW インデックスを構築する。
// DOCS の埋め込みは OPFS にキャッシュし、あれば再埋め込みをスキップする(一番重い処理)。
const buildDatabase = async (
  embedder: Embedder,
  cacheKey: string
): Promise<{ db: duckdb.AsyncDuckDB; dim: number; source: "cache" | "computed" }> => {
  // まず OPFS キャッシュを試す。なければ埋め込みを計算して保存する。
  let docVectors = await readEmbeddingsCache(cacheKey);
  let source: "cache" | "computed";
  if (docVectors && docVectors.length === DOCS.length) {
    source = "cache";
    console.log("DOCS の埋め込みを OPFS キャッシュから復元");
  } else {
    docVectors = [];
    for (const doc of DOCS) {
      docVectors.push(await embedder.embed(doc, "passage"));
    }
    await writeEmbeddingsCache(cacheKey, docVectors);
    source = "computed";
    console.log("DOCS の埋め込みを計算して OPFS に保存");
  }

  const dim = docVectors[0]?.length ?? embedder.dim;
  if (!dim) throw new Error("埋め込み次元の判定に失敗しました");

  const worker = new duckdb_worker();
  const logger = new duckdb.VoidLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(duckdb_wasm);
  await db.open({});

  const conn = await db.connect();
  await conn.query("INSTALL vss;");
  await conn.query("LOAD vss;");

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
    "cache" | "computed" | null
  >(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">(
    "loading"
  );
  const [resultRows, setResultRows] = useState<StructRowProxy<any>[]>([]);
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
          cacheKeyFor(selectedModelId)
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
      const newResultRows: StructRowProxy<any>[] = newResults
        .toArray()
        .map((row: any) => JSON.parse(row));

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
