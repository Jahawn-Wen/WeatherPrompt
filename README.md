# WeatherPrompt


**WeatherPrompt** is a **training-free, all-weather, text-guided cross-view localization framework**. The core idea is to introduce **open-ended weather descriptions** into the retrieval process to counteract the observational shifts and cross-view misalignments caused by rain, fog, snow, and night conditions.

We introduce a **training-free multi-weather prompting pipeline**: leveraging University-1652 and SUES-200, we synthesize adverse-weather views and use a **two-stage prompt (“weather-first, then spatial”)** to express **coarse-to-fine semantics**, thereby **auto-generating high-quality open-set image–text pairs** for alignment and supervision.

Extensive experiments validate the effectiveness of WeatherPrompt. On University-1652, Drone $\rightarrow$ Satellite achieves R@1 77.14\% / AP 80.20\% and Satellite $\rightarrow$ Drone achieves R@1 87.72\% / AP 76.39\%. Compared with representative methods, Drone $\rightarrow$ Satellite improves R@1 by 11.99\% and AP by 11.04\%, while Satellite $\rightarrow$ Drone improves R@1 by 3.04\% and AP by 10.64\%. On SUES-200, Satellite $\rightarrow$ Drone reaches R@1 80.73\% / AP 66.12\%, showing consistent advantages. Under real-world Dark+Rain+Fog videos, Drone $\rightarrow$ Satellite attains AP 64.94\% and Satellite $\rightarrow$ Drone AP 72.15\%, evidencing strong generalization and robustness in adverse weather.


## News
* The **Code** is coming soon. Welcome to communicate！
