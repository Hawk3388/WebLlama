# WebLlama

WebLlama is a web search extension for [Ollama](https://github.com/ollama/ollama), providing enhanced search capabilities, even for smaller models.

## Features

- Perform web searches using the [DuckDuckGo](https://github.com/deedy5/duckduckgo_search) API
- Integrate with [Ollama](https://github.com/ollama/ollama) for chat-based interactions
- Support various search and retrieval configurations

## Requirements

- **Python**: Version 3.10 or higher (≥ 3.10)  
- **Ollama**: If not installed, download it from [here](https://ollama.com/download)

## Installation

To install WebLlama, download the .whl file from the [releases page](https://github.com/Hawk3388/WebLlama/releases) and install it via:

```sh
cd downloads
pip install WebLlama-1.0.0-py3-none-any.whl
```

if you encounter an error during the installation make sure you have wheel installed, if not install it with:

```sh
pip install wheel
```

to confirm the installation run:

```sh
webllama --version
```

## Usage

WebLlama provides a command-line interface for interacting with the tool. Here are some of the available commands:

```sh
webllama [flags]
webllama [command]
```

### Available Commands

- **serve**: Start Ollama
- **create**: Create a model from a Modelfile
- **show**: Show information for a model
- **run**: Run a model
- **stop**: Stop a running model
- **pull**: Pull a model from a registry
- **push**: Push a model to a registry
- **list**: List all models
- **ps**: List running models
- **cp**: Copy a model
- **rm**: Remove a model
- **help**: Help about any command

### Flags

- **-h, --help**: Show help for WebLlama
- **-v, --version**: Show version information

### Example

To run a model with WebLlama:

```sh
webllama run <model_name>
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

[Hawk3388](https://github.com/Hawk3388)

## Contributing

Contributions are welcome! Please open an [issue](https://github.com/Hawk3388/WebLlama/issues) or submit a [pull request](https://github.com/Hawk3388/WebLlama/pulls) on GitHub.

## Acknowledgements

- [Ollama](https://github.com/ollama/ollama)
- [LangChain](https://github.com/langchain-ai/langchain)
- [duckduckgo_search](https://github.com/deedy5/duckduckgo_search)
