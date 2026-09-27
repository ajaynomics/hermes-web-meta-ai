# hermes-web-meta-ai

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) web search backend that runs
Hermes' `web_search` tool through the **server-side `web_search` built-in on Meta's Responses API**
(Muse Spark). Muse Spark searches and opens pages on Meta's side, then returns structured
`{title, url, description}` rows, so results look the same as Tavily, Exa or any other Hermes web provider.

It is search-only. `web_extract` keeps whatever extract backend you already use.

It follows the bundled xAI web backend (`plugins/web/xai`), but uses only the public
`agent.web_search_provider` plugin surface. No core changes are needed.

## Install

Requires Hermes Agent >= 0.21.3 (tested on 0.21.3 and 0.21.5).

```bash
hermes plugins install ajaynomics/hermes-web-meta-ai --enable
```

Or copy this directory to `$HERMES_HOME/plugins/web-meta-ai/` and run `hermes plugins enable web-meta-ai`.
Plugins are per profile: install and enable it in every profile that should use it.

## Configure

```yaml
web:
  search_backend: meta-ai        # search only
  meta-ai:                       # all optional
    model: muse-spark-1.3        # any Muse Spark model your key can use
    base_url: https://api.meta.ai/v1
    timeout: 120                 # seconds
```

Auth uses the same env vars as Hermes' bundled `meta-ai` model provider: `MODEL_API_KEY`
(or `META_API_KEY` / `META_MODEL_API_KEY`). `META_BASE_URL` overrides the base URL.

If the plugin is not loaded, Hermes' resolver ignores `search_backend: meta-ai` and falls back to its
normal provider walk.

## Cost and latency

Each search is one extra Responses call. Meta bills web search per query, on top of tokens.
In testing, a search took about 20–30 s and used 15–35k input tokens, because Muse Spark opens
pages to check its results.

## Trust model

Like the xAI backend, the model writes the titles and descriptions and chooses the URLs, and the
query text can steer it. Treat returned URLs as model-generated links.

## Tests

From a Hermes install (the Docker image works):

```bash
PYTHONPATH=/opt/hermes python -m unittest discover -s tests -v
```

## License

MIT
