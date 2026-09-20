# Result tables (auto-generated)


## Task 2

| run | method | source_mean_accuracy | source_mean_macro_f1 | target_accuracy | target_macro_f1 | domain_separability | source_photo_accuracy | source_photo_macro_f1 | source_art_painting_accuracy | source_art_painting_macro_f1 | source_cartoon_accuracy | source_cartoon_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2_source_only | source_only | 0.9382 | 0.9375 | 0.6747 | 0.6578 | 0.9986 | 0.9763 | 0.9706 | 0.8978 | 0.8970 | 0.9406 | 0.9449 |
| t2_dan | dan | 0.9342 | 0.9320 | 0.7753 | 0.7382 | 0.8948 | 0.9644 | 0.9575 | 0.9124 | 0.9127 | 0.9257 | 0.9257 |
| t2_dann | dann | 0.9358 | 0.9357 | 0.6320 | 0.6431 | 1.0000 | 0.9792 | 0.9748 | 0.9002 | 0.8995 | 0.9278 | 0.9327 |
| t2_cdan | cdan | 0.9279 | 0.9268 | 0.3156 | 0.4316 | 0.9959 | 0.9733 | 0.9675 | 0.8783 | 0.8763 | 0.9321 | 0.9367 |
| t2_dan_lambda01 | dan | 0.9232 | 0.9231 | 0.7376 | 0.7533 | 0.9918 | 0.9644 | 0.9578 | 0.8881 | 0.8910 | 0.9172 | 0.9205 |
| t2_dan_lambda10 | dan | 0.2172 | 0.0639 | 0.0356 | 0.0185 | 0.8702 | 0.2611 | 0.0647 | 0.2141 | 0.0711 | 0.1762 | 0.0559 |

## Task 3

| method | run | source_mean_accuracy | source_mean_macro_f1 | source_worst_macro_f1 | target_accuracy | target_macro_f1 | source_domain_separability | sharpness_delta | source_photo_macro_f1 | source_art_painting_macro_f1 | source_cartoon_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ERM | t2_source_only | None | None | None | 0.6747 | 0.6578 | 0.8618 | 0.2726 | 0.9706 | 0.8970 | 0.9449 |
| DAN-DG | t3_dan_dg | 0.9183 | 0.9123 | 0.8926 | 0.7137 | 0.7048 | 0.6776 | 134.2262 | 0.9285 | 0.8926 | 0.9159 |
| SAM | t3_sam | 0.9589 | 0.9583 | 0.9372 | 0.7137 | 0.7301 | 0.8553 | 0.1662 | 0.9707 | 0.9372 | 0.9671 |
| t3_sam_rho001 (study) | t3_sam_rho001 | 0.9492 | 0.9458 | 0.9305 | 0.6747 | 0.7098 | 0.8520 | 0.2163 |  |  |  |
| t3_sam_rho01 (study) | t3_sam_rho01 | 0.9411 | 0.9389 | 0.9173 | 0.7035 | 0.7111 | 0.8158 | 0.1180 |  |  |  |
