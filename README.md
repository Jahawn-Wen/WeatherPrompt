# WeatherPrompt


**WeatherPrompt** is a **training-free, all-weather, text-guided cross-view localization framework**. The core idea is to introduce **open-ended weather descriptions** into the retrieval process to counteract the observational shifts and cross-view misalignments caused by rain, fog, snow, and night conditions.

We introduce a **training-free multi-weather prompting pipeline**: leveraging University-1652 and SUES-200, we synthesize adverse-weather views and use a **two-stage prompt (“weather-first, then spatial”)** to express **coarse-to-fine semantics**, thereby **auto-generating high-quality open-set image–text pairs** for alignment and supervision.
