# WebLlama

WebLlama is a web search extension for [Ollama](https://github.com/ollama/ollama), providing enhanced search capabilities, even for small models.

## Features

- Perform web searches using the [DuckDuckGo](https://github.com/deedy5/duckduckgo_search) API
- Integrate with [Ollama](https://github.com/ollama/ollama) for chat-based interactions
- Support various search and retrieval configurations

## Installation

To install WebLlama, download the .whl file from the [releases page](https://github.com/Hawk3388/webllama/releases) and install it via:

```sh
cd downloads
pip install webllama-1.0.0-py3-none-any.whl
```

if you encounter an error make sure you are on python >= 3.10 and you have wheel installed, if not install it:

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

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

[Hawk3388](https://github.com/Hawk3388)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Acknowledgements

- [Ollama](https://github.com/ollama/ollama)
- [LangChain](https://github.com/langchain-ai/langchain)
- [duckduckgo_search](https://github.com/deedy5/duckduckgo_search)
