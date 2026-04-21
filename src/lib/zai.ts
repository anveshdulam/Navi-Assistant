import ZAI from "z-ai-web-dev-sdk";

export type ZAIClient = Awaited<ReturnType<typeof ZAI.create>>;

let zaiInstance: ZAIClient | null = null;
let zaiInitPromise: Promise<ZAIClient> | null = null;

const buildZaiConfig = () => {
  const apiKey = process.env.ZAI_API_KEY ?? process.env.OPENAI_API_KEY;
  const baseURL = process.env.ZAI_BASE_URL ?? process.env.OPENAI_BASE_URL;
  const config: Record<string, string> = {};

  if (apiKey) config.apiKey = apiKey;
  if (baseURL) {
    config.baseURL = baseURL;
    config.baseUrl = baseURL;
  }

  return Object.keys(config).length > 0 ? config : undefined;
};

export const getZAI = async (): Promise<ZAIClient> => {
  if (zaiInstance) return zaiInstance;
  if (!zaiInitPromise) {
    const config = buildZaiConfig();
    zaiInitPromise = ZAI.create(
      config as unknown as Record<string, string> | undefined
    )
      .then((client) => {
        zaiInstance = client;
        return client;
      })
      .catch((error) => {
        zaiInitPromise = null;
        throw error;
      });
  }
  return zaiInitPromise;
};
