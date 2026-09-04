# Demo dataset / 演示数据 / 데모 데이터

This directory contains a fully synthetic dataset for trying Matching Clothes. It contains no production photos, customer data, supplier names, or private paths.

本目录是一套完全合成的试用数据，不包含真实业务照片、顾客信息、供应商名称或私人路径。

이 폴더는 Matching Clothes를 시험하기 위한 완전 합성 데이터입니다. 실제 업무 사진, 고객 정보, 공급업체 이름 또는 개인 경로가 포함되어 있지 않습니다.

## Try it / 试用 / 사용 방법

1. Use `style_library/` as the style library / 将 `style_library/` 选为款式图库 / `style_library/`를 상품 이미지 라이브러리로 선택합니다.
2. Build or update the index / 建立或更新索引 / 인덱스를 생성하거나 업데이트합니다.
3. Import `store_photos/` / 导入 `store_photos/` / `store_photos/`를 가져옵니다.
4. Analyze and compare the top candidate with `expected_matches.csv` / 分析后与 `expected_matches.csv` 对照 / 분석 결과를 `expected_matches.csv`와 비교합니다.

Run `python generate_demo.py` to regenerate the same deterministic dataset.
