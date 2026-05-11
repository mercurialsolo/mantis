"""search_results recipe — extract organic search results from a SERP.

Navigates to a search-engine results page (Google in the shipped plan)
and pulls the top N organic results as ``{title, url, snippet}`` rows.
Schema is declared inline in the plan's claude step.
"""
