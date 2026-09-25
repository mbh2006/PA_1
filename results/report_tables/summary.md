# Result tables (auto-generated)


## Task 1 - clean / interventions (per backbone)

| condition | backbone | accuracy | macro_f1 | mean_max_confidence | prediction_consistency | cosine_stability |
|---|---|---|---|---|---|---|
| grayscale | resnet50 | 0.9680 | 0.9682 | 0.8895 | 0.9680 | 0.8070 |
| grayscale | vit_b16 | 0.9600 | 0.9600 | 0.9376 | 0.9660 | 0.7411 |
| grayscale | clip_vit_b32 | 0.9540 | 0.9535 | 0.1958 | 0.9640 | 0.8833 |
| grayscale | clip_vit_b32_zero_shot | 0.9280 | 0.9260 | 0.9240 | 0.9460 | 0.8833 |
| hue | resnet50 | 0.9400 | 0.9402 | 0.8735 | 0.9420 | 0.7627 |
| hue | vit_b16 | 0.9680 | 0.9679 | 0.9221 | 0.9700 | 0.7124 |
| hue | clip_vit_b32 | 0.9560 | 0.9560 | 0.1876 | 0.9580 | 0.8821 |
| hue | clip_vit_b32_zero_shot | 0.9300 | 0.9284 | 0.9200 | 0.9480 | 0.8821 |
| shuffle | resnet50 | 0.9000 | 0.9018 | 0.8657 | 0.9080 | 0.6941 |
| shuffle | vit_b16 | 0.9120 | 0.9124 | 0.8511 | 0.9120 | 0.6331 |
| shuffle | clip_vit_b32 | 0.7980 | 0.7983 | 0.1535 | 0.7960 | 0.7520 |
| shuffle | clip_vit_b32_zero_shot | 0.7820 | 0.7826 | 0.7669 | 0.7800 | 0.7520 |
| translate_left_8 | resnet50 | 0.9780 | 0.9781 | 0.8808 | 0.9840 | 0.9724 |
| translate_left_8 | vit_b16 | 0.9740 | 0.9740 | 0.9437 | 0.9860 | 0.9483 |
| translate_left_8 | clip_vit_b32 | 0.9660 | 0.9659 | 0.2056 | 0.9700 | 0.9779 |
| translate_left_8 | clip_vit_b32_zero_shot | 0.9300 | 0.9277 | 0.9343 | 0.9600 | 0.9779 |
| translate_right_8 | resnet50 | 0.9800 | 0.9801 | 0.8798 | 0.9880 | 0.9673 |
| translate_right_8 | vit_b16 | 0.9780 | 0.9780 | 0.9427 | 0.9880 | 0.9487 |
| translate_right_8 | clip_vit_b32 | 0.9800 | 0.9800 | 0.2059 | 0.9760 | 0.9773 |
| translate_right_8 | clip_vit_b32_zero_shot | 0.9360 | 0.9340 | 0.9333 | 0.9580 | 0.9773 |
| translate_up_8 | resnet50 | 0.9780 | 0.9780 | 0.8836 | 0.9840 | 0.9665 |
| translate_up_8 | vit_b16 | 0.9840 | 0.9840 | 0.9463 | 0.9840 | 0.9411 |
| translate_up_8 | clip_vit_b32 | 0.9780 | 0.9778 | 0.2062 | 0.9880 | 0.9770 |
| translate_up_8 | clip_vit_b32_zero_shot | 0.9460 | 0.9447 | 0.9349 | 0.9620 | 0.9770 |
| translate_down_8 | resnet50 | 0.9740 | 0.9741 | 0.8828 | 0.9900 | 0.9661 |
| translate_down_8 | vit_b16 | 0.9880 | 0.9880 | 0.9464 | 0.9880 | 0.9432 |
| translate_down_8 | clip_vit_b32 | 0.9660 | 0.9658 | 0.2065 | 0.9660 | 0.9771 |
| translate_down_8 | clip_vit_b32_zero_shot | 0.9380 | 0.9357 | 0.9338 | 0.9700 | 0.9771 |
| translate_left_16 | resnet50 | 0.9720 | 0.9722 | 0.8781 | 0.9900 | 0.9581 |
| translate_left_16 | vit_b16 | 0.9820 | 0.9820 | 0.9435 | 0.9980 | 0.9760 |
| translate_left_16 | clip_vit_b32 | 0.9560 | 0.9557 | 0.2018 | 0.9580 | 0.9599 |
| translate_left_16 | clip_vit_b32_zero_shot | 0.9320 | 0.9301 | 0.9280 | 0.9540 | 0.9599 |
| translate_right_16 | resnet50 | 0.9780 | 0.9779 | 0.8739 | 0.9920 | 0.9460 |
| translate_right_16 | vit_b16 | 0.9760 | 0.9759 | 0.9440 | 0.9920 | 0.9750 |
| translate_right_16 | clip_vit_b32 | 0.9620 | 0.9619 | 0.2023 | 0.9660 | 0.9622 |
| translate_right_16 | clip_vit_b32_zero_shot | 0.9280 | 0.9268 | 0.9253 | 0.9560 | 0.9622 |
| translate_up_16 | resnet50 | 0.9800 | 0.9800 | 0.8847 | 0.9940 | 0.9540 |
| translate_up_16 | vit_b16 | 0.9800 | 0.9800 | 0.9457 | 0.9900 | 0.9768 |
| translate_up_16 | clip_vit_b32 | 0.9640 | 0.9639 | 0.2031 | 0.9660 | 0.9643 |
| translate_up_16 | clip_vit_b32_zero_shot | 0.9440 | 0.9425 | 0.9318 | 0.9600 | 0.9643 |
| translate_down_16 | resnet50 | 0.9780 | 0.9781 | 0.8807 | 0.9940 | 0.9506 |
| translate_down_16 | vit_b16 | 0.9800 | 0.9800 | 0.9471 | 0.9940 | 0.9755 |
| translate_down_16 | clip_vit_b32 | 0.9640 | 0.9640 | 0.2041 | 0.9660 | 0.9663 |
| translate_down_16 | clip_vit_b32_zero_shot | 0.9400 | 0.9386 | 0.9318 | 0.9600 | 0.9663 |
| translate_left_32 | resnet50 | 0.9640 | 0.9640 | 0.8725 | 0.9700 | 0.9331 |
| translate_left_32 | vit_b16 | 0.9800 | 0.9801 | 0.9374 | 0.9940 | 0.9460 |
| translate_left_32 | clip_vit_b32 | 0.9520 | 0.9518 | 0.1977 | 0.9600 | 0.9583 |
| translate_left_32 | clip_vit_b32_zero_shot | 0.9260 | 0.9245 | 0.9123 | 0.9500 | 0.9583 |
| translate_right_32 | resnet50 | 0.9660 | 0.9659 | 0.8700 | 0.9760 | 0.9229 |
| translate_right_32 | vit_b16 | 0.9700 | 0.9699 | 0.9392 | 0.9860 | 0.9468 |
| translate_right_32 | clip_vit_b32 | 0.9580 | 0.9573 | 0.1983 | 0.9720 | 0.9605 |
| translate_right_32 | clip_vit_b32_zero_shot | 0.9160 | 0.9126 | 0.9193 | 0.9540 | 0.9605 |
| translate_up_32 | resnet50 | 0.9720 | 0.9721 | 0.8764 | 0.9900 | 0.9342 |
| translate_up_32 | vit_b16 | 0.9780 | 0.9780 | 0.9382 | 0.9880 | 0.9408 |
| translate_up_32 | clip_vit_b32 | 0.9580 | 0.9574 | 0.2003 | 0.9720 | 0.9729 |
| translate_up_32 | clip_vit_b32_zero_shot | 0.9260 | 0.9229 | 0.9196 | 0.9560 | 0.9729 |
| translate_down_32 | resnet50 | 0.9800 | 0.9799 | 0.8742 | 0.9840 | 0.9310 |
| translate_down_32 | vit_b16 | 0.9760 | 0.9760 | 0.9428 | 0.9900 | 0.9393 |
| translate_down_32 | clip_vit_b32 | 0.9560 | 0.9555 | 0.1992 | 0.9740 | 0.9720 |
| translate_down_32 | clip_vit_b32_zero_shot | 0.9280 | 0.9250 | 0.9224 | 0.9700 | 0.9720 |
| clean | resnet50 | 0.9820 | 0.9820 | 0.8841 | 1.0000 | 1.0000 |
| clean | vit_b16 | 0.9840 | 0.9840 | 0.9484 | 1.0000 | 1.0000 |
| clean | clip_vit_b32 | 0.9760 | 0.9757 | 0.2073 | 1.0000 | 1.0000 |
| clean | clip_vit_b32_zero_shot | 0.9480 | 0.9459 | 0.9344 | 1.0000 | 1.0000 |

## Task 1 - cue conflicts (shape bias + coverage)

| backbone | predictor | cosine_stability | shape | texture | other | total | shape_bias | coverage |
|---|---|---|---|---|---|---|---|---|
| resnet50 | head | 0.5713 | 200 | 29 | 36 | 265 | 87.3362 | 86.4151 |
| vit_b16 | head | 0.5875 | 245 | 5 | 15 | 265 | 98.0000 | 94.3396 |
| clip_vit_b32 | head | 0.7980 | 225 | 13 | 27 | 265 | 94.5378 | 89.8113 |
| clip_vit_b32 | zero_shot | 0.7980 | 220 | 19 | 26 | 265 | 92.0502 | 90.1887 |

## Task 2 - main comparison and lambda_MMD study

| run | method | source_mean_accuracy | source_mean_macro_f1 | target_accuracy | target_macro_f1 | domain_separability | source_photo_accuracy | source_photo_macro_f1 | source_art_painting_accuracy | source_art_painting_macro_f1 | source_cartoon_accuracy | source_cartoon_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2_source_only | source_only | 0.9382 | 0.9375 | 0.6747 | 0.6578 | 0.9986 | 0.9763 | 0.9706 | 0.8978 | 0.8970 | 0.9406 | 0.9449 |
| t2_dan | dan | 0.9342 | 0.9320 | 0.7753 | 0.7382 | 0.8948 | 0.9644 | 0.9575 | 0.9124 | 0.9127 | 0.9257 | 0.9257 |
| t2_dann | dann | 0.9358 | 0.9357 | 0.6320 | 0.6431 | 1.0000 | 0.9792 | 0.9748 | 0.9002 | 0.8995 | 0.9278 | 0.9327 |
| t2_cdan | cdan | 0.9279 | 0.9268 | 0.3156 | 0.4316 | 0.9959 | 0.9733 | 0.9675 | 0.8783 | 0.8763 | 0.9321 | 0.9367 |
| t2_dan_lambda01 | dan | 0.9232 | 0.9231 | 0.7376 | 0.7533 | 0.9918 | 0.9644 | 0.9578 | 0.8881 | 0.8910 | 0.9172 | 0.9205 |
| t2_dan_lambda10 | dan | 0.2172 | 0.0639 | 0.0356 | 0.0185 | 0.8702 | 0.2611 | 0.0647 | 0.2141 | 0.0711 | 0.1762 | 0.0559 |

## Task 3 - main comparison and rho study

| method | run | source_mean_accuracy | source_mean_macro_f1 | source_worst_macro_f1 | target_accuracy | target_macro_f1 | source_domain_separability | sharpness_delta | source_photo_macro_f1 | source_art_painting_macro_f1 | source_cartoon_macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ERM | t2_source_only | 0.9382 | 0.9375 | 0.8970 | 0.6747 | 0.6578 | 0.8618 | 0.2726 | 0.9706 | 0.8970 | 0.9449 |
| DAN-DG | t3_dan_dg | 0.9183 | 0.9123 | 0.8926 | 0.7137 | 0.7048 | 0.6776 | 134.2262 | 0.9285 | 0.8926 | 0.9159 |
| SAM | t3_sam | 0.9589 | 0.9583 | 0.9372 | 0.7137 | 0.7301 | 0.8553 | 0.1662 | 0.9707 | 0.9372 | 0.9671 |
| t3_sam_rho001 (study) | t3_sam_rho001 | 0.9492 | 0.9458 | 0.9305 | 0.6747 | 0.7098 | 0.8520 | 0.2163 |  |  |  |
| t3_sam_rho01 (study) | t3_sam_rho01 | 0.9411 | 0.9389 | 0.9173 | 0.7035 | 0.7111 | 0.8158 | 0.1180 |  |  |  |

## Task 4 - post-hoc scores on the vanilla model

| score | auroc_near | auroc_far | auroc_all_unknowns | threshold | known_test_acceptance | near_rejection | near_fpr95 | far_rejection | far_fpr95 |
|---|---|---|---|---|---|---|---|---|---|
| MSP | 0.8087 | 0.8912 | 0.8499 | 0.1431 | 0.9428 | 0.2737 | 0.7263 | 0.4612 | 0.5388 |
| MLS | 0.7871 | 0.8842 | 0.8356 | -5.8048 | 0.9475 | 0.3000 | 0.7000 | 0.5312 | 0.4688 |
| Energy | 0.7875 | 0.8848 | 0.8362 | -5.9605 | 0.9496 | 0.2963 | 0.7037 | 0.5325 | 0.4675 |
| Mahalanobis | 0.7859 | 0.9155 | 0.8507 | 2944.7837 | 0.9493 | 0.2362 | 0.7638 | 0.4875 | 0.5125 |

## Task 4 - model comparison (MLS + PROSER placeholder)

| model | score | closed_set_accuracy | auroc_near | auroc_far | auroc_all_unknowns | threshold | known_test_acceptance | near_rejection | near_fpr95 | far_rejection | far_fpr95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla | MLS | 0.9446 | 0.7871 | 0.8842 | 0.8356 | -5.8048 | 0.9475 | 0.3000 | 0.7000 | 0.5312 | 0.4688 |
| GCSC | MLS | 0.9503 | 0.8176 | 0.9122 | 0.8649 | -6.1117 | 0.9477 | 0.3563 | 0.6438 | 0.5775 | 0.4225 |
| PROSER | MLS | 0.9408 | 0.7640 | 0.8538 | 0.8089 | -4.6622 | 0.9487 | 0.2600 | 0.7400 | 0.4662 | 0.5337 |
| PROSER | Placeholder | 0.9408 | 0.7541 | 0.8724 | 0.8133 | 0.0000 | 0.9492 | 0.2625 | 0.7375 | 0.5000 | 0.5000 |
