# Result tables (auto-generated)


## Task 2

| run | method | source_mean_accuracy | source_mean_macro_f1 | target_accuracy | target_macro_f1 | domain_separability | source_photo_accuracy | source_photo_macro_f1 | source_art_painting_accuracy | source_art_painting_macro_f1 | source_cartoon_accuracy | source_cartoon_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2_source_only | source_only | 0.9382 | 0.9375 | 0.6747 | 0.6578 | 0.9986 | 0.9763 | 0.9706 | 0.8978 | 0.8970 | 0.9406 | 0.9449 |
| t2_dan | dan | 0.9528 | 0.9515 | 0.6880 | 0.6225 | 0.8374 | 0.9763 | 0.9719 | 0.9416 | 0.9419 | 0.9406 | 0.9407 |
| t2_dann | dann | 0.9358 | 0.9357 | 0.6320 | 0.6431 | 1.0000 | 0.9792 | 0.9748 | 0.9002 | 0.8995 | 0.9278 | 0.9327 |
| t2_cdan | cdan | 0.9279 | 0.9268 | 0.3156 | 0.4316 | 0.9959 | 0.9733 | 0.9675 | 0.8783 | 0.8763 | 0.9321 | 0.9367 |

## Task 3

| method | run | source_mean_accuracy | source_mean_macro_f1 | source_worst_macro_f1 | target_accuracy | target_macro_f1 | source_domain_separability | sharpness_delta | source_photo_macro_f1 | source_art_painting_macro_f1 | source_cartoon_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ERM | t2_source_only | None | None | None | 0.6747 | 0.6578 | None | None | 0.9706 | 0.8970 | 0.9449 |
| SAM | t3_sam | 0.9589 | 0.9583 | 0.9372 | 0.7137 | 0.7301 | 0.8553 | 0.1662 | 0.9707 | 0.9372 | 0.9671 |
