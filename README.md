# LLM Token-Budget Allocator

A tool that decides how to allocate a fixed token/dollar budget across a set of competing LLM tasks to maximize total value, comparing an exact dynamic-programming solution against a greedy approximation.

Install and run: `pip install -e .` then `allocator run --budget 20 --tasks tasks.json`.
