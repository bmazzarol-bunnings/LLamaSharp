using BenchmarkDotNet.Attributes;
using LLama.Common;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace LLama.Benchmark;

[MemoryDiagnoser]
[SimpleJob]
public class LlamaEmbedderBenchmarks
{
    private LLamaEmbedder _embedder = null!;
    private List<string> _batch1 = null!;
    private List<string> _batch10 = null!;
    private List<string> _batch50 = null!;

    [GlobalSetup]
    public void Setup()
    {
        // Initialize the embedder with the embedding model
        var modelParams = new ModelParams(Constants.EmbeddingModelPath)
        {
            ContextSize = 2048,
        };

        var weights = LLamaWeights.LoadFromFile(modelParams);
        _embedder = new LLamaEmbedder(weights, modelParams);

        // Prepare test batches with varied content
        _batch1 = new List<string>
        {
            "This is a sample sentence for embedding computation."
        };

        _batch10 = new List<string>
        {
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming our world in unprecedented ways.",
            "Machine learning algorithms require large amounts of data to train effectively.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models can recognize patterns in complex datasets.",
            "Text embeddings capture semantic meaning in high-dimensional vector spaces.",
            "Large language models demonstrate remarkable capabilities across various domains.",
            "Vector databases store and retrieve embeddings efficiently at scale.",
            "Semantic search leverages embeddings to find relevant information.",
            "Transformer architectures have revolutionized natural language understanding."
        };

        _batch50 = new List<string>();
        for (int i = 0; i < 50; i++)
        {
            _batch50.Add($"This is test sentence number {i + 1} for batch embedding evaluation. " +
                        $"It contains enough text to make the embedding computation meaningful and realistic. " +
                        $"The content varies slightly to ensure diverse embedding vectors are generated.");
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _embedder?.Dispose();
    }

    [Benchmark]
    public async Task EmbedBatch1()
    {
        foreach (var text in _batch1)
        {
            await _embedder.GetEmbeddings(text);
        }
    }

    [Benchmark]
    public async Task EmbedBatch10()
    {
        foreach (var text in _batch10)
        {
            await _embedder.GetEmbeddings(text);
        }
    }

    [Benchmark]
    public async Task EmbedBatch50()
    {
        foreach (var text in _batch50)
        {
            await _embedder.GetEmbeddings(text);
        }
    }
}