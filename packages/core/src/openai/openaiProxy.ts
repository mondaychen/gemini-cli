/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';

import { ContentGenerator } from '../core/contentGenerator.js';
import {
  GenerateContentParameters,
  GenerateContentResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  FunctionCall,
  Part,
} from '@google/genai';
import { encoding_for_model, TiktokenModel } from 'tiktoken';

import {
  toOpenAIMessages,
  toOpenAIRequestBody,
  completionToGenerateContentResponse,
} from './converter.js';

export interface OpenAIConfig {
  model: string;
  apiKey: string;
  baseURL: string;
}

export interface HttpOptions {
  /** Additional HTTP headers to be sent with the request. */
  headers?: Record<string, string>;
}

export class OpenAIProxy implements ContentGenerator {
  private readonly openai: OpenAI;

  constructor(
    private readonly config: OpenAIConfig,
    private readonly httpOptions: HttpOptions,
  ) {
    this.openai = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
      defaultHeaders: httpOptions.headers,
    });
  }

  async generateContent(
    request: GenerateContentParameters,
  ): Promise<GenerateContentResponse> {
    const requestBody: OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming =
      {
        ...toOpenAIRequestBody(request),
        stream: false,
      };
    const response = await this.openai.chat.completions.create(requestBody);

    return completionToGenerateContentResponse(response);
  }

  async generateContentStream(
    request: GenerateContentParameters,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const requestBody: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming =
      {
        ...toOpenAIRequestBody(request),
        stream: true,
      };
    const stream = await this.openai.chat.completions.create(requestBody);
    const functionCall: FunctionCall = {
      id: '',
      name: '',
    };
    let functionArgsString = ''; // use a separate string since FunctionCall.args expects an object
    return (async function* () {
      for await (const chunk of stream) {
        const msg = chunk.choices[0];
        const tool = msg.delta.tool_calls?.[0];
        if (tool) {
          if (tool.id) {
            functionCall.id = tool.id;
          }
          if (tool.function?.name) {
            functionCall.name += tool.function.name;
          }
          if (tool.function?.arguments) {
            functionArgsString += tool.function.arguments;
          }
        }
        const functionCalls = [];
        // only add function call if the finish reason is tool_calls or function_call
        // at this point, the argument string is complete
        if (
          msg.finish_reason === 'tool_calls' ||
          msg.finish_reason === 'function_call'
        ) {
          try {
            if (functionArgsString.length > 0) {
              functionCall.args = JSON.parse(functionArgsString) as Record<
                string,
                unknown
              >;
            }
            functionCalls.push(functionCall);
          } catch (error) {
            console.error('Error parsing tool call arguments:', error, functionArgsString);
          }
        }
        const parts: Part[] = [];
        if (functionCalls.length > 0) {
          parts.push({
            functionCall: functionCalls[0],
          });
        } else {
          parts.push({
            text: msg.delta.content || '',
          });
        }
        const response = {
          candidates: [
            {
              content: {
                parts,
                role: 'model',
              },
            },
          ],
          text: msg.delta.content || '',
          data: undefined,
          functionCalls,
          executableCode: '',
          codeExecutionResult: undefined,
        };
        yield response;
      }
    })();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    const encoding = encoding_for_model(this.config.model as TiktokenModel);
    const messages = toOpenAIMessages(request.contents);
    const tokens = encoding.encode(
      messages.map((message) => message.content).join('\n'),
    );
    return {
      totalTokens: tokens.length,
    };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw Error('Not implemented');
  }
}
