import { OpenAI, AzureOpenAI } from 'openai';

export class Chat {
  private openai: OpenAI | AzureOpenAI;
  private isAzure: boolean;
  private isGithubModels: boolean;

  constructor(apikey: string) {
    this.isAzure = Boolean(
        process.env.AZURE_API_VERSION && process.env.AZURE_DEPLOYMENT,
    );

    this.isGithubModels = process.env.USE_GITHUB_MODELS === 'true';

    if (this.isAzure) {
      // Azure OpenAI configuration
      this.openai = new AzureOpenAI({
        apiKey: apikey,
        endpoint: process.env.OPENAI_API_ENDPOINT || '',
        apiVersion: process.env.AZURE_API_VERSION || '',
        deployment: process.env.AZURE_DEPLOYMENT || '',
      });
    } else {
      // Standard OpenAI configuration
      this.openai = new OpenAI({
        apiKey: apikey,
        baseURL: this.isGithubModels ? 'https://models.github.ai/inference' : process.env.OPENAI_API_ENDPOINT || 'https://api.openai.com/v1',
      });
    }
  }

  private generatePrompt = (patch: string) => {
    const answerLanguage = process.env.LANGUAGE
        ? `Answer me in ${process.env.LANGUAGE},`
        : '';

    // MNN 专用的前置提示词
    const mnnSystemPrompt = `你是 MNN (Mobile Neural Network) 项目的代码审查助手。MNN 是阿里巴巴开发的轻量级深度学习推理框架。

在代码审查时请重点关注以下方面：
1. **性能优化**：MNN 专为移动端高性能设计。请关注潜在的性能瓶颈、不必要的内存分配或低效算法。
2. **跨平台兼容性**：确保代码能在不同平台（iOS、Android、Linux、Windows、macOS）上正常运行。
3. **内存管理**：特别关注内存泄漏、缓冲区溢出和资源释放问题。
4. **API 一致性**：确保新代码遵循 MNN 现有的 API 风格和命名规范。
5. **错误处理**：验证边界情况的错误处理和优雅降级。
6. **线程安全**：检查多线程代码中的竞态条件和同步问题。

`;

    const userPrompt = process.env.PROMPT || '请审查以下代码变更，重点关注潜在的 bug、风险和改进建议。';
    
    const jsonFormatRequirement = '\n请按以下 JSON 格式返回你的审查意见：\n' +
        '{\n' +
        '  "reviews": [\n' +
        '    {\n' +
        '      "hunk_header": string, // 代码块的 @@ 行号标记（如 "@@ -10,5 +10,7 @@"），可选\n' +
        '      "lgtm": boolean, // 如果这段代码看起来没问题则为 true，有需要改进的地方则为 false\n' +
        '      "review_comment": string // 详细的审查意见，支持 markdown 格式。如果 lgtm 为 true 则为空字符串\n' +
        '    }\n' +
        '  ]\n' +
        '}\n' +
        '请分别审查每个代码块（以 @@ 标记），对需要改进的代码块给出反馈。\n' +
        '确保返回的是有效的 JSON 对象，包含 reviews 数组。\n';

    return `${mnnSystemPrompt}${userPrompt}${jsonFormatRequirement} ${answerLanguage}:
    ${patch}
    `;
  };

  public codeReview = async (patch: string): Promise<Array<{ lgtm: boolean, review_comment: string, hunk_header?: string }> | { lgtm: boolean, review_comment: string, hunk_header?: string }> => {
    if (!patch) {
      return {
        lgtm: true,
        review_comment: ""
      };
    }

    console.time('code-review cost');
    const prompt = this.generatePrompt(patch);

    const res = await this.openai.chat.completions.create({
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      model: process.env.MODEL || (this.isGithubModels ? 'openai/gpt-4o-mini' : 'gpt-4o-mini'),
      temperature: +(process.env.temperature || 0) || 1,
      top_p: +(process.env.top_p || 0) || 1,
      max_tokens: process.env.max_tokens ? +process.env.max_tokens : undefined,
      response_format: {
        type: "json_object"
      },
    });

    console.timeEnd('code-review cost');

    if (res.choices.length) {
      try {
        const json = JSON.parse(res.choices[0].message.content || "");
        // If response has a 'reviews' array, return it directly
        if (json.reviews && Array.isArray(json.reviews)) {
          return json.reviews;
        }
        // Otherwise, treat as a single review response
        return json;
      } catch (e) {
        return {
          lgtm: false,
          hunk_header: patch.split('\n')[0].startsWith('@@') ? patch.split('\n')[0] : undefined,
          review_comment: res.choices[0].message.content || ""
        }
      }
    }

    return {
      lgtm: true,
      review_comment: ""
    }
  };
}
