import json
import pickle
import torch
import os
import re

def replace_parentheses_pattern(text):
    """(영어)/(한글) -> 영어 형태로 변환"""
    pattern = r'\(([a-zA-Z\-]+)\)/\([^)]+\)'
    return re.sub(pattern, r'\1', text)


def convert_to_github_format(pkl_file, bias_json_file, output_dir):
    """
    우리가 만든 파일들을 GitHub 형태로 변환 (수정 버전)
    """
    # 기존 pkl 파일 로드
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # 기존 bias JSON 로드
    with open(bias_json_file, 'r', encoding='utf-8') as f:
        bias_data = json.load(f)

    os.makedirs(os.path.join(output_dir, "fbanks"), exist_ok=True)

    github_format = {}

    # features와 bias_data 매핑
    for i, (feature, text, major) in enumerate(zip(data['features'], data['texts'], data['majors'])):
        file_id = f"korean_lecture_{i:06d}"  # korean_lecture_000000 형태

        # 텍스트 정규화 (동료가 언급한 부분)
        text = replace_parentheses_pattern(text)

        # pt 파일로 저장
        fbank_filename = f"{file_id}_fbank.pt"
        fbank_path = os.path.join(output_dir, "fbanks", fbank_filename)
        torch.save(feature, fbank_path)

        # bias_words 찾기
        blist = []
        utt_key = f"utterance_{i}"
        if utt_key in bias_data:
            blist = bias_data[utt_key].get('bias_words', [])

        # GitHub 형태로 변환
        github_format[file_id] = {
            "fbank": fbank_path,
            "words": text,
            "blist": blist
        }

        # 진행상황 출력
        if (i + 1) % 10000 == 0:
            print(f"처리됨: {i + 1} / {len(data['features'])}")

    # 최종 JSON 저장
    with open(os.path.join(output_dir, "korean_lecture_bias.json"), "w", encoding='utf-8') as f:
        json.dump(github_format, f, ensure_ascii=False, indent=4)

    print(f"변환 완료! {len(github_format)}개 파일 처리됨")
    print(f"샘플 텍스트: {list(github_format.values())[0]['words']}")


def check_pkl_structure(pkl_file):
    """pkl 파일 구조 확인"""
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    print("PKL 파일 키들:", data.keys())
    if 'features' in data:
        print("features 개수:", len(data['features']))
    if 'texts' in data:
        print("texts 개수:", len(data['texts']))
        print("첫 번째 텍스트:", data['texts'][0])

if __name__ == '__main__':

    # check_pkl_structure("/Users/rover0811/PycharmProjects/WhisperTCPGen/korean_processed/korean_lecture_features.pkl")

    convert_to_github_format(
        "/Users/rover0811/PycharmProjects/WhisperTCPGen/korean_processed/korean_lecture_features.pkl",
        "/Users/rover0811/PycharmProjects/WhisperTCPGen/korean_processed/korean_lecture_bias.json",
        "./results"
    )