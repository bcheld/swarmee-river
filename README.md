<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Swarmee River
  </h1>

  <h2>
    Enterprise analytics + coding assistant built on the Strands Agents SDK.
  </h2>

  <div align="center">
    <a href="https://github.com/strands-agents/agent-builder/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/agent-builder"/></a>
    <a href="https://github.com/strands-agents/agent-builder/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/agent-builder"/></a>
    <a href="https://github.com/strands-agents/agent-builder/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/agent-builder"/></a>
    <a href="https://github.com/strands-agents/agent-builder/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/agent-builder"/></a>
    <a href="https://pypi.org/project/swarmee-river/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/swarmee-river"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/swarmee-river"/></a>
  </div>
  
  <p>
    <a href="https://strandsagents.com/">Documentation</a>
    ◆ <a href="https://github.com/strands-agents/samples">Samples</a>
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
    ◆ <a href="https://github.com/strands-agents/mcp-server">MCP Server</a>
  </p>
</div>

Swarmee River is an interactive, enterprise-oriented analytics + coding assistant built on the Strands Agents SDK. It keeps what works (simple packaging, hot-loaded tools, AWS connectivity) while adding better context management, SOPs, and observability.

## Quick Start

```bash
# Install
pipx install swarmee-river

# Run interactive mode
swarmee

# Build a custom tool and use it immediately
swarmee "Create a tool named sentiment_analyzer that analyzes text sentiment and test it with some examples"

# Pipe content to build an agent based on specifications
cat agent-spec.txt | swarmee "Build a specialized agent based on these specifications"

# Use with knowledge base to extend existing tools
swarmee --kb YOUR_KB_ID "Load my previous calculator tool and enhance it with scientific functions"
```

## Features

- 🏗️ Create and test custom tools with instant hot-reloading
- 🤖 Build specialized agents with focused capabilities
- 🔄 Extend existing tools and enhance their functionality
- 💬 Interactive command-line interface with rich output
- ⛔ Interrupt a running agent with `Esc`
- 🛠️ Powerful integrated tools (12+ tools including shell, editor, HTTP, Python)
- 🧠 Knowledge base integration for persisting and loading tools
- 🎮 Customizable system prompt for specialized agents
- 🪄 Nested agent capabilities with tool delegation
- 🔧 Dynamic tool loading for extending functionality
- 🖥️ Environment variable management and customization

## Integrated Tools

Swarmee River uses Strands Tools and supports hot-loading tools from `./tools`.

- **agent_graph**: Create and manage graphs of agents
- **project_context**: Explore the project in the current working directory (files/search/git status)
- **calculator**: Perform mathematical operations
- **cron**: Task scheduling with cron jobs *(not available on Windows)*
- **current_time**: Get the current date and time
- **editor**: File editing operations like line edits, search, and undo
- **environment**: Manage environment variables
- **generate_image**: Create AI generated images with Amazon Bedrock
- **http_request**: Make API calls, fetch web data, and call local HTTP servers
- **image_reader**: Process and analyze images
- **journal**: Create structured tasks and logs for agents to manage and work from
- **load_tool**: Dynamically load more tools at runtime
- **memory**: Agent memory persistence in Amazon Bedrock Knowledge Bases
- **nova_reels**: Create AI generated videos with Nova Reels on Amazon Bedrock
- **python_repl**: Run Python code *(not available on Windows)*
- **retrieve**: Semantically retrieve data from Amazon Bedrock Knowledge Bases for RAG, memory, and other purposes
- **shell**: Execute shell commands *(not available on Windows)*
- **slack**: Slack integration with real-time events, API access, and message sending
- **speak**: Generate speech from text using macOS say command or Amazon Polly
- **stop**: Force stop the agent event loop
- **store_in_kb**: Save content to knowledge bases for future reference
- **strand**: Create nested agent instances with specialized capabilities
- **swarm**: Coordinate multiple AI agents in a swarm / network of agents
- **think**: Perform deep thinking by creating parallel branches of agentic reasoning
- **use_aws**: Interact with AWS services
- **use_llm**: Run a new AI event loop with custom prompts
- **welcome**: Manage the Swarmee welcome text
- **workflow**: Orchestrate sequenced workflows

## Jupyter Notebook Integration

Swarmee River can be used inside Jupyter via an IPython extension that registers a `%%swarmee` cell magic.

1) Install optional deps:
```bash
pip install -e .[jupyter]
```

2) In a notebook:
```python
%load_ext swarmee_river.jupyter

%%swarmee
Review this notebook and suggest improvements to the code.
```

Example notebook: `examples/notebooks/swarmee_magic_demo.ipynb`

## Knowledge Base Integration

Swarmee River can leverage Amazon Bedrock Knowledge Bases to store and retrieve useful context and artifacts.

### Set up your Knowledge Base

#### Prerequisites
- AWS account with IAM user 
- Access to Amazon Bedrock console
- Permissions to create IAM roles and S3 buckets

#### Console Setup (Recommended)

1. **Access Amazon Bedrock Console**
   - Sign in to [AWS Management Console](https://console.aws.amazon.com/bedrock)
   - Navigate to **Knowledge bases** in the left panel

2. **Create Knowledge Base**
   - Click **Create** → **Knowledge base with vector store**
   - Enter a name and description for your knowledge base

3. **Configure IAM Role**
   - Choose **Create and use a new service role** (recommended)
   - Or select an existing role with Bedrock permissions

4. **Set Up Data Source**
   - Choose your data source type, select `Custom` for Strands Agent
   - Configure connection details

5. **Configure Embeddings**
   - Select an embeddings model (e.g., Amazon Titan Text Embeddings V2)
   - Choose embedding type: `float32` (precise) or `binary` (cost-effective)

6. **Choose Vector Store**
   - **Quick create** (recommended): Let Bedrock create and manage the vector store
   - **Custom**: Use existing OpenSearch Serverless, Aurora PostgreSQL, or S3 Vectors

7. **Review and Create**
   - Review all configurations
   - Click **Create knowledge base**
   - Wait for creation to complete (status: "Ready" or "Available")

8. **Sync Data Source**
   - Select your knowledge base
   - Click **Sync** in the data source section
   - Wait for initial sync to complete

#### API Setup (Advanced)

Create via AWS CLI or SDK:

```bash
# Example using AWS CLI
aws bedrock-agent create-knowledge-base \
  --name "MyKnowledgeBase" \
  --description "Swarmee River KB" \
  --role-arn "arn:aws:iam::ACCOUNT:role/AmazonBedrockExecutionRoleForKnowledgeBase" \
  --knowledge-base-configuration '{
    "type": "VECTOR",
    "vectorKnowledgeBaseConfiguration": {
      "embeddingModelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }' \
  --storage-configuration '{
    "type": "OPENSEARCH_SERVERLESS",
    "opensearchServerlessConfiguration": {
      "collectionArn": "arn:aws:aoss:us-east-1:ACCOUNT:collection/YOUR_COLLECTION",
      "vectorIndexName": "bedrock-knowledge-base-index",
      "fieldMapping": {
        "vectorField": "bedrock-knowledge-base-default-vector",
        "textField": "AMAZON_BEDROCK_TEXT_CHUNK",
        "metadataField": "AMAZON_BEDROCK_METADATA"
      }
    }
  }'
```
For more set up details, please see [AWS Bedrock Knowledgebase](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-create.html)

#### Get Your Knowledge Base ID

After creation, find your Knowledge Base ID:
- **Console**: Copy from the knowledge base details page
- **CLI**: `aws bedrock-agent list-knowledge-bases`

The ID format: `ABCDEFGHIJ` (10 characters)

### Use your KnowledgeBase

```bash
# Load and extend tools from your knowledge base
swarmee --kb YOUR_KB_ID "Load my data_visualizer tool and add 3D plotting capabilities"

# Or set a default knowledge base via environment variable
export STRANDS_KNOWLEDGE_BASE_ID="YOUR_KB_ID"
swarmee "Find my most recent agent configuration and make it more efficient"
```

Features:
- 🔄 Retrieve previously created tools and agent configurations
- 💾 Persistent storage for your custom tools and agents
- 🛠️ Ability to iteratively improve tools across sessions
- 🔍 Find and extend tools built in previous sessions

## Model Configuration

### Optimized Defaults

Swarmee River defaults to an optimized Bedrock configuration when using `--model-provider bedrock`:

```json
{
    "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
    "max_tokens": 32767,
    "boto_client_config": {
        "read_timeout": 900,
        "connect_timeout": 900,
        "retries": {
            "max_attempts": 3,
            "mode": "adaptive"
        }
    },
    "additional_request_fields": {
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2048
        }
    }
}
```

These settings provide:
- Claude Sonnet 4 (latest high-performance model)
- Maximum token output (32,768 tokens)
- Extended timeouts (15 minutes) for complex operations
- Automatic retries with adaptive backoff
- Interleaved thinking capability for real-time reasoning during responses
- Enabled thinking capability with 2,048 token budget for recursive reasoning

You can customize these values using environment variables:

```bash
# Maximum tokens for responses
export STRANDS_MAX_TOKENS=32000

# Budget for agent thinking/reasoning
export STRANDS_BUDGET_TOKENS=1024
```

## Custom Model Provider

You can configure Swarmee to use a different model provider with specific settings by passing in the following arguments:

```bash
swarmee --model-provider <NAME> --model-config <JSON|FILE>
```

As an example, if you wanted to use the packaged Ollama provider with a specific model id, you would run:

```bash
swarmee --model-provider ollama --model-config '{"model_id": "<ID>"}'
```

Swarmee River is packaged with `bedrock`, `ollama`, and `openai`.

### OpenAI (low-cost default)

Create a local `.env` file with your key:

```bash
echo "OPENAI_API_KEY=..." > .env
echo "OPENAI_BASE_URL=https://api.openai.com/v1" >> .env
```

Then run:

```bash
swarmee --model-provider openai "Hello from gpt-5-nano"
```

Tip: if you see a `max_tokens` / output token limit error, increase the output cap:

```bash
swarmee --model-provider openai --max-output-tokens 1024 "List your tools (briefly)"
```

To override the model/limits:

```bash
swarmee --model-provider openai --model-config '{"model_id":"gpt-5-nano","params":{"max_completion_tokens":256}}' "Summarize this repo"
```

If you have implemented a custom model provider ([instructions](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/)) and would like to use it with Swarmee, create a python module under the directory "$CWD/.models" and expose an `instance` function that returns an instance of your provider. As an example, assume you have:

```bash
$ cat ./.models/custom_model.py
from mymodels import CustomModel

def instance(**config):
    return CustomModel(**config)
```

You can then use it with Swarmee by running:

```bash
$ swarmee --model-provider custom_model --model-config <JSON|FILE>
```

## Custom System Prompts

```bash
# Via environment variable
export SWARMEE_SYSTEM_PROMPT="You are a Python expert."

# Or local file
echo "You are a security expert." > .prompt
```

## 🌍 Environment Variables Configuration

Swarmee River provides customization through environment variables (Strands variables still work where applicable):

| Environment Variable | Description | Default | 
|----------------------|-------------|---------|
| STRANDS_MODEL_ID | Claude model ID to use for inference | us.anthropic.claude-sonnet-4-20250514-v1:0 |
| STRANDS_MAX_TOKENS | Maximum tokens for agent responses | 32768 |
| STRANDS_BUDGET_TOKENS | Token budget for agent thinking/reasoning | 2048 |
| STRANDS_THINKING_TYPE | Type of thinking capability | enabled |
| STRANDS_ANTHROPIC_BETA | Anthropic beta features (comma-separated) | interleaved-thinking-2025-05-14 |
| STRANDS_CACHE_TOOLS | Tool caching strategy | default |
| STRANDS_CACHE_PROMPT | Prompt caching strategy | default |
| STRANDS_SYSTEM_PROMPT | Custom system prompt (overrides .prompt file) | None |
| STRANDS_KNOWLEDGE_BASE_ID | Default Knowledge Base ID | None |
| STRANDS_TOOL_CONSOLE_MODE | Enable rich console UI | enabled |
| BYPASS_TOOL_CONSENT | Skip tool confirmation prompts | false |

Swarmee-specific variables (selected):

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| SWARMEE_SYSTEM_PROMPT | Custom system prompt | None |
| SWARMEE_KNOWLEDGE_BASE_ID | Default Knowledge Base ID | None |
| SWARMEE_CONTEXT_MANAGER | `summarize` / `sliding` / `none` | summarize |
| SWARMEE_CONTEXT_BUDGET_TOKENS | Prompt budget before summarization (approx) | 20000 |
| SWARMEE_ENABLE_SOPS | Allow only these SOPs (comma-separated) | None |
| SWARMEE_DISABLE_SOPS | Block these SOPs (comma-separated) | None |
| SWARMEE_LOG_EVENTS | Enable JSONL event logging | true |
| SWARMEE_LOG_S3_BUCKET | Optional S3 bucket for log uploads | None |

## Exit

Type `exit`, `quit`, or press `Ctrl+C`/`Ctrl+D`

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
