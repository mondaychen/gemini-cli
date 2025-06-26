/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import OpenAI from 'openai';
import {
  GenerateContentParameters,
  GenerateContentResponse,
  ContentListUnion,
  Content,
  PartUnion,
  Schema,
  ToolListUnion,
  Tool,
  FunctionCall,
} from '@google/genai';

type GenminiRole = 'user' | 'model' | 'system';
type SimpleOpenAIRole = 'user' | 'assistant' | 'developer';

function isContent(content: ContentListUnion): content is Content {
  return (
    typeof content === 'object' &&
    !Array.isArray(content) &&
    'parts' in content &&
    Array.isArray(content.parts)
  );
}

function toOpenAIRole(genminiRole: GenminiRole): SimpleOpenAIRole {
  if (genminiRole === 'model') {
    return 'assistant';
  }
  if (genminiRole === 'system') {
    return 'developer';
  }
  return 'user';
}

function toOpenAIMessage(
  content: PartUnion,
  defaultRole: SimpleOpenAIRole = 'user',
):
  | OpenAI.Chat.Completions.ChatCompletionMessageParam
  | OpenAI.Chat.Completions.ChatCompletionToolMessageParam {
  if (typeof content === 'string') {
    return {
      role: defaultRole,
      content,
    };
  }
  // convert "Part", when there's function call
  if (
    typeof content === 'object' &&
    'functionCall' in content &&
    typeof content.functionCall === 'object'
  ) {
    return {
      role: 'assistant',
      content: null,
      tool_calls: [
        {
          id: content.functionCall.id || 'unknown_tool_id',
          type: 'function',
          function: {
            name: content.functionCall.name || 'unknown_tool_name',
            arguments: JSON.stringify(content.functionCall.args),
          },
        },
      ],
    };
  }

  // convert "Part", when there's function response
  if (
    typeof content === 'object' &&
    'functionResponse' in content &&
    typeof content.functionResponse === 'object'
  ) {
    return {
      role: 'tool',
      content: JSON.stringify(content.functionResponse.response, null, 2) || '',
      tool_call_id: content.functionResponse.id || '',
    };
  }
  // convert "Part",when there's text
  if (
    typeof content === 'object' &&
    'text' in content &&
    typeof content.text === 'string'
  ) {
    return {
      role: defaultRole,
      content: content.text,
    };
  }

  throw new Error(`Unsupported content type: ${typeof content}`);
}

function contentToOpenAIMessages(
  content: Content,
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  return (
    content.parts?.map((part) =>
      toOpenAIMessage(part, toOpenAIRole(content.role as GenminiRole)),
    ) || []
  );
}

export function toOpenAIMessages(
  contents: ContentListUnion,
  defaultRole: SimpleOpenAIRole = 'user',
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  if (isContent(contents)) {
    return contentToOpenAIMessages(contents);
  }
  if (Array.isArray(contents)) {
    return contents.flatMap((content) => {
      // list of "Content"
      if (isContent(content)) {
        return contentToOpenAIMessages(content);
      }
      // list of "Part"
      return toOpenAIMessage(content, defaultRole);
    });
  }
  // single "Part"
  return [toOpenAIMessage(contents, defaultRole)];
}

export function completionToGenerateContentResponse(
  response: OpenAI.Chat.Completions.ChatCompletion,
): GenerateContentResponse {
  const tools = response.choices[0].message.tool_calls;
  let functionCalls: FunctionCall[] = [];
  try {
    functionCalls =
      tools?.map((tool) => ({
        id: tool.id,
        name: tool.function.name,
        args: JSON.parse(tool.function.arguments) as Record<string, unknown>,
      })) || [];
  } catch (error) {
    console.error('Error parsing tool call arguments:', error);
  }

  return {
    candidates: response.choices.map((choice) => ({
      content: {
        parts: [
          {
            text: choice.message.content || '',
          },
        ],
        role: 'model',
      },
    })),
    text: response.choices[0].message.content || '',
    data: undefined,
    functionCalls: functionCalls || [],
    executableCode: '',
    codeExecutionResult: undefined,
  };
}

// deep search in this schemaObject, find all "type" properties, and convert them to lowercase
function recursivelyConvertType(schema: Schema): Record<string, unknown> {
  const schemaObject = schema as Record<string, unknown>;
  Object.keys(schemaObject).forEach((key) => {
    if (key === 'type' && typeof schemaObject[key] === 'string') {
      schemaObject[key] = schemaObject[key].toLowerCase();
    } else if (schemaObject[key] && typeof schemaObject[key] === 'object') {
      schemaObject[key] = recursivelyConvertType(schemaObject[key]);
    }
  });
  return schemaObject;
}

export function toOpenAISchemaFormat(
  schema: Schema,
): OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming['response_format'] {
  // deep search in this schemaObject, find all "type" properties, and convert them to lowercase
  const schemaObject = recursivelyConvertType(schema);
  return {
    type: 'json_schema',
    json_schema: {
      name: 'object', // required by OpenAI
      schema: schemaObject,
    },
  };
}

export function toOpenAIFunctionCall(
  tools: ToolListUnion,
): OpenAI.Chat.Completions.ChatCompletionCreateParamsNonStreaming['tools'] {
  // ignore callable tools for now
  const nonCallableTools = tools.filter(
    (tool) => !('callTool' in tool),
  ) as Tool[];
  const toolDeclarations = nonCallableTools
    .flatMap((tool) => tool.functionDeclarations)
    .filter((toolDeclaration) => toolDeclaration != null);
  return toolDeclarations.map((toolDeclaration) => ({
    type: 'function',
    function: {
      name: toolDeclaration.name!,
      description: toolDeclaration.description,
      parameters: recursivelyConvertType(toolDeclaration.parameters!),
    },
  }));
}

export function toOpenAIRequestBody(
  request: GenerateContentParameters,
): OpenAI.Chat.Completions.ChatCompletionCreateParams {
  const requestBody: OpenAI.Chat.Completions.ChatCompletionCreateParams = {
    model: request.model,
    messages: toOpenAIMessages(request.contents),
  };
  // handle system instruction
  if (request.config?.systemInstruction) {
    const systemInstruction = toOpenAIMessages(
      request.config.systemInstruction,
      'developer',
    );
    requestBody.messages = [...systemInstruction, ...requestBody.messages];
  }
  if (request.config?.temperature) {
    requestBody.temperature = request.config.temperature;
  }
  if (request.config?.topP) {
    requestBody.top_p = request.config?.topP;
  }
  if (
    request.config?.responseMimeType === 'application/json' &&
    request.config?.responseSchema
  ) {
    requestBody.response_format = toOpenAISchemaFormat(
      request.config.responseSchema,
    );
  }
  if (request.config?.tools) {
    requestBody.tools = toOpenAIFunctionCall(request.config.tools);
  }
  return requestBody;
}
