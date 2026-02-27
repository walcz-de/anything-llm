const { toChunks, maximumChunkLength } = require("../../helpers");

class LocalAiEmbedder {
  constructor() {
    if (!process.env.EMBEDDING_BASE_PATH)
      throw new Error("No embedding base path was set.");
    if (!process.env.EMBEDDING_MODEL_PREF)
      throw new Error("No embedding model was set.");

    this.className = "LocalAiEmbedder";
    const { OpenAI: OpenAIApi } = require("openai");
    this.model = process.env.EMBEDDING_MODEL_PREF;
    this.openai = new OpenAIApi({
      baseURL: process.env.EMBEDDING_BASE_PATH,
      apiKey: process.env.LOCAL_AI_API_KEY ?? null,
    });

    // Limit of how many strings we can process in a single pass to stay with resource or network limits
    this.maxConcurrentChunks = 50;
    this.embeddingMaxChunkLength = maximumChunkLength();

    this.log(
      `Initialized with ${this.model} - Max Size: ${this.embeddingMaxChunkLength}` +
        (this.outputDimensions
          ? ` - Output Dimensions: ${this.outputDimensions}`
          : " Assuming default output dimensions")
    );
    this.log(`TRACE: LocalAiEmbedder constructor. model=${this.model}, outputDimensions=${this.outputDimensions}, process.env.EMBEDDING_OUTPUT_DIMENSIONS=${process.env.EMBEDDING_OUTPUT_DIMENSIONS}`);
  }

  log(text, ...args) {
    console.log(`\x1b[36m[${this.className}]\x1b[0m ${text}`, ...args);
  }

  get outputDimensions() {
    if (
      process.env.EMBEDDING_OUTPUT_DIMENSIONS &&
      !isNaN(process.env.EMBEDDING_OUTPUT_DIMENSIONS) &&
      process.env.EMBEDDING_OUTPUT_DIMENSIONS > 0
    )
      return parseInt(process.env.EMBEDDING_OUTPUT_DIMENSIONS);
    return null;
  }

  async embedTextInput(textInput) {
    const result = await this.embedChunks(
      Array.isArray(textInput) ? textInput : [textInput]
    );
    return result?.[0] || [];
  }

  async embedChunks(textChunks = []) {
    this.log(`TRACE: LocalAiEmbedder.embedChunks start. Chunks count: ${textChunks.length}, outputDimensions=${this.outputDimensions}`);
    const embeddingRequests = [];
    for (const chunk of toChunks(textChunks, this.maxConcurrentChunks)) {
      embeddingRequests.push(
        new Promise(async (resolve) => {
          try {
            const body = {
              model: this.model,
              input: chunk,
            };

            if (this.outputDimensions) {
              body.dimensions = this.outputDimensions;
            }

            this.log(`TRACE: Sending fetch to ${this.openai.baseURL}/embeddings`.replace(/\/+$/, "").replace(/\/embeddings\/embeddings$/, "/embeddings"));
            const response = await fetch(`${this.openai.baseURL}/embeddings`.replace(/\/+$/, "").replace(/\/embeddings\/embeddings$/, "/embeddings"), {
              method: "POST",
              headers: {
                "Accept": "application/json",
                "Content-Type": "application/json",
                ...(this.openai.apiKey
                  ? { Authorization: `Bearer ${this.openai.apiKey}` }
                  : {}),
              },
              body: JSON.stringify(body),
            });

            if (!response.ok) {
              const errorData = await response.json().catch(() => ({}));
              throw new Error(errorData.error?.message || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            if (result?.data?.[0]?.embedding) {
              this.log(`TRACE: LocalAiEmbedder chunk result dimension: ${result.data[0].embedding.length}`);
            }
            resolve({ data: result?.data, error: null });
          } catch (e) {
            this.log(`TRACE: LocalAiEmbedder error: ${e.message}`);
            e.type = "failed_to_embed";
            resolve({ data: [], error: e });
          }
        })
      );
    }

    const { data = [], error = null } = await Promise.all(
      embeddingRequests
    ).then((results) => {
      // If any errors were returned from LocalAI abort the entire sequence because the embeddings
      // will be incomplete.
      const errors = results
        .filter((res) => !!res.error)
        .map((res) => res.error)
        .flat();
      if (errors.length > 0) {
        let uniqueErrors = new Set();
        errors.map((error) =>
          uniqueErrors.add(`[${error.type}]: ${error.message}`)
        );

        return {
          data: [],
          error: Array.from(uniqueErrors).join(", "),
        };
      }
      return {
        data: results.map((res) => res?.data || []).flat(),
        error: null,
      };
    });

    if (!!error) throw new Error(`LocalAI Failed to embed: ${error}`);
    return data.length > 0 &&
      data.every((embd) => embd.hasOwnProperty("embedding"))
      ? data.map((embd) => embd.embedding)
      : null;
  }
}

module.exports = {
  LocalAiEmbedder,
};
