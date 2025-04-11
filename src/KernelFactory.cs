using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.Extensions.Logging;
using Microsoft.SemanticKernel.Connectors.Google;
using System;

namespace Workshop.SemanticKernel.MultiAgent {

    public static class KernelFactory
    {
        public static Kernel CreateKernel(ILoggerFactory loggerFactory, Settings settings, string model, TransformerBackend backend)
        {
            var backendSettings = settings.GetTransformerBackendSettings(backend);
            var kernelBuilder = Kernel.CreateBuilder();
            kernelBuilder.Services.AddSingleton(loggerFactory);
            switch (backend)
            {
                case TransformerBackend.OpenAI:
                    var oaiSettings = backendSettings as Settings.OpenAISettings;
                    // Add null check for oaiSettings to avoid null reference exception
                    var organization = oaiSettings?.Organization ?? string.Empty;
                    return kernelBuilder.AddOpenAIChatCompletion(model, backendSettings.ApiKey, organization).Build();
                case TransformerBackend.AzureOpenAI:
                    return kernelBuilder.AddAzureOpenAIChatCompletion(model, backendSettings.Endpoint, backendSettings.ApiKey).Build();
                case TransformerBackend.Ollama:
                    return kernelBuilder.AddOllamaChatCompletion(model, new Uri(backendSettings.Endpoint)).Build(); // If you have an api key, add it as the 3rd parameter
                case TransformerBackend.Gemini:
                    return kernelBuilder.AddGoogleAIGeminiChatCompletion(model, backendSettings.ApiKey).Build();
                case TransformerBackend.Perplexity:
                    // Return a default implementation since Perplexity is not supported yet
                    // Remove this case once Perplexity is properly supported
                    throw new NotImplementedException("Perplexity support is not yet implemented");
                default:
                    throw new ArgumentOutOfRangeException(nameof(backend), backend, null);
            }
        }
        public static PromptExecutionSettings GetExecutionSettings(TransformerBackend backend)
        {
            switch (backend)
            {
                case TransformerBackend.OpenAI:
                case TransformerBackend.AzureOpenAI:
                    return new OpenAIPromptExecutionSettings
                    {
                        // Use FunctionChoiceBehavior for modern SK versions
                        FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
                        // OR if using older versions or needing specific OpenAI features:
                        // ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
                    };
                case TransformerBackend.Perplexity:
                    // Return a default execution settings for Perplexity
                    return new PromptExecutionSettings
                    {
                        FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
                    };
                case TransformerBackend.Gemini:
                    return new GeminiPromptExecutionSettings
                    {
                        ToolCallBehavior = GeminiToolCallBehavior.AutoInvokeKernelFunctions
                    };
                case TransformerBackend.Ollama:
                default:
                    return new PromptExecutionSettings()
                    {
                        FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
                    };
            }
        }
        
        public static TransformerBackend ConvertFrom(string backend) 
        {
            return backend.ToLower() switch
            {
                "openai" => TransformerBackend.OpenAI,
                "azureopenai" => TransformerBackend.AzureOpenAI,
                "ollama" => TransformerBackend.Ollama,
                "gemini" => TransformerBackend.Gemini,
                "perplexity" => TransformerBackend.Perplexity,
                _ => throw new ArgumentOutOfRangeException(nameof(backend), backend, null)
            };
        }
    }

}