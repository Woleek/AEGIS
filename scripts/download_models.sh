# AdaFace
uv run gdown "https://drive.google.com/uc?id=1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI" -O ./models/adaface_ir50_ms1mv2.ckpt

# Head Pose Estimation FSANet
wget -O ./models/fsanet-1x1-iter-688590.onnx https://raw.githubusercontent.com/omasaht/headpose-fsanet-pytorch/master/pretrained/fsanet-1x1-iter-688590.onnx
wget -O ./models/fsanet-var-iter-688590.onnx https://raw.githubusercontent.com/omasaht/headpose-fsanet-pytorch/master/pretrained/fsanet-var-iter-688590.onnx

# ArcFace
curl 'https://my.microsoftpersonalcontent.com/personal/4a83b6b633b029cc/_layouts/15/download.aspx?UniqueId=33b029cc-b6b6-2083-804a-431600000000&Translate=false&tempauth=v1e.eyJzaXRlaWQiOiIyNGFkYTQzZC00NzUyLTRlMmQtODE0Ny0xNzYxYmRkNzJiNmMiLCJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvbXkubWljcm9zb2Z0cGVyc29uYWxjb250ZW50LmNvbUA5MTg4MDQwZC02YzY3LTRjNWItYjExMi0zNmEzMDRiNjZkYWQiLCJleHAiOiIxNzYxNTk4Nzg2In0.XyzgtXNnJXxY4Y5Q3IrDYUulXEeBh9dctyw9UhKGjOqQFqw-GXKA7_wPIuUeXkQei0KM-Lku4b-1RLGo2fGm71ruchWo4UBr-wgWe-2rCsnuVzZuqlZhpZvxLA2NwmoNkZYsvece9pWlME1UwYWxw-E8dwQCA_J5m5VE_RiQz7nPXBEyihqpsY2CrFitfwcgGo4vXlA7zblSCkArCg--AX8UU-eBfNwpM-3QFfihegsF9m1Ddn7syQEG909mPL1RUOnM-1z7yj1M2X3qypf3dk6kS-U64vArB4HeCnPsQqOFqWEJRllQ7gKs4SSlI_xGKxU4E6vMeOmiMqR1mxe2BzEoWHOSzq_IFUvfnvHB5jKG87_5M4xj02s6dYuDuKa72acxOCIphzrw80q759UX4xamqJpNwSlWKEpckrZAUKx1MB_aVBGgW7eB9WNoELlGb8qfACJq7igfwndIxaPrva0hgvaAVyheSv_eKWrZwrB0wWPZy-6mRTOk5VNTu4gidUPIAhBaLlVIOFQvtNoGYw.OifdM3mtvHiwaU7J3iCIeBYSTquS4mIdkPMRrE6QCLY&ApiVersion=2.0' \
  -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8' \
  -H 'accept-language: pl-PL,pl;q=0.9' \
  -H 'priority: u=0, i' \
  -H 'referer: https://onedrive.live.com/' \
  -H 'sec-ch-ua: "Brave";v="141", "Not?A_Brand";v="8", "Chromium";v="141"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "Windows"' \
  -H 'sec-fetch-dest: iframe' \
  -H 'sec-fetch-mode: navigate' \
  -H 'sec-fetch-site: cross-site' \
  -H 'sec-fetch-storage-access: none' \
  -H 'sec-fetch-user: ?1' \
  -H 'sec-gpc: 1' \
  -H 'upgrade-insecure-requests: 1' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36' \
  --output ./models/arcface_ir50_ms1mv3.pth