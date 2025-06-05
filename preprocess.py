import json
import os
import whisper
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import time


def process_single_file(args):
    """
    단일 파일 처리 함수 (멀티프로세싱용)
    """
    json_file, training_dir = args

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 음성 파일 경로와 텍스트 추출
        audio_relative_path = data["01_dataset"]["3_src_path"]  # source/eng/comp/C02199/U00001.wav
        audio_file_path = '/'.join(audio_relative_path.split('/')[-2:])

        text = data["06_transcription"]["1_text"]
        major = data["03_lectureinfo"]["3_major_category"]
        file_id = data["01_dataset"]["1_identifier"]

        # Label 폴더에서 Voice 폴더로 경로 변경
        # Training/Label/xxx.json -> Training/Voice/source/eng/comp/C02199/U00001.wav
        voice_path = os.path.join(training_dir, "Voice", audio_file_path)

        if os.path.exists(voice_path):
            # Whisper 입력 형태로 변환
            audio = whisper.load_audio(voice_path)
            mel = whisper.log_mel_spectrogram(audio)

            return {
                'feature': mel,
                'text': text,
                'major': major,
                'file_id': file_id,
                'status': 'success'
            }
        else:
            return {'status': 'audio_not_found', 'file': str(json_file), 'expected_path': voice_path}

    except Exception as e:
        return {'status': 'error', 'file': str(json_file), 'error': str(e)}


def dump_feature_korean_mp(training_dir, output_dir, num_workers=None):
    """
    멀티프로세싱을 사용한 한국어 대학 강의 데이터 전처리
    training_dir: IT_Lecture/Training 경로
    """
    if num_workers is None:
        num_workers = cpu_count()  # 최대 8개 프로세스

    print(f"Using {num_workers} workers for processing")

    os.makedirs(output_dir, exist_ok=True)

    # Label 폴더에서 JSON 파일 목록 수집
    label_dir = os.path.join(training_dir, "Label")
    json_files = list(Path(label_dir).glob("**/*.json"))
    print(f"Found {len(json_files)} JSON files in {label_dir}")

    # 멀티프로세싱 인자 준비 (training_dir 전달)
    args_list = [(json_file, training_dir) for json_file in json_files]

    features = []
    texts = []
    majors = []
    file_ids = []

    start_time = time.time()

    # 청크 단위로 처리 (메모리 절약)
    chunk_size = 100
    total_processed = 0

    for i in range(0, len(args_list), chunk_size):
        chunk = args_list[i:i + chunk_size]

        with Pool(num_workers) as pool:
            results = pool.map(process_single_file, chunk)

        # 결과 처리
        for result in results:
            if result['status'] == 'success':
                features.append(result['feature'])
                texts.append(result['text'])
                majors.append(result['major'])
                file_ids.append(result['file_id'])
                total_processed += 1
            elif result['status'] == 'audio_not_found':
                print(f"Audio not found: {result['file']}")
                print(f"Expected at: {result['expected_path']}")
            else:
                print(f"Error processing {result['file']}: {result['error']}")

        # 진행상황 출력
        elapsed = time.time() - start_time
        print(f"Processed {total_processed}/{len(json_files)} files ({elapsed:.1f}s)")

        # 중간 저장 (메모리 절약)
        if len(features) >= 1000:
            save_chunk_data(features, texts, majors, file_ids, output_dir, i // chunk_size)
            features, texts, majors, file_ids = [], [], [], []

    # 마지막 데이터 저장
    if features:
        save_chunk_data(features, texts, majors, file_ids, output_dir, "final")

    # 모든 청크 병합
    merge_chunk_files(output_dir)

    print(f"Total processed: {total_processed} samples in {time.time() - start_time:.1f}s")
    return total_processed


def save_chunk_data(features, texts, majors, file_ids, output_dir, chunk_id):
    """청크 데이터 저장"""
    save_data = {
        'features': features,
        'texts': texts,
        'majors': majors,
        'file_ids': file_ids
    }

    chunk_file = os.path.join(output_dir, f'chunk_{chunk_id}.pkl')
    with open(chunk_file, 'wb') as f:
        pickle.dump(save_data, f)


def merge_chunk_files(output_dir):
    """청크 파일들을 하나로 병합"""
    all_features = []
    all_texts = []
    all_majors = []
    all_file_ids = []

    chunk_files = list(Path(output_dir).glob("chunk_*.pkl"))

    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            data = pickle.load(f)

        all_features.extend(data['features'])
        all_texts.extend(data['texts'])
        all_majors.extend(data['majors'])
        all_file_ids.extend(data['file_ids'])

        # 청크 파일 삭제
        os.remove(chunk_file)

    # 최종 파일 저장
    final_data = {
        'features': all_features,
        'texts': all_texts,
        'majors': all_majors,
        'file_ids': all_file_ids
    }

    with open(os.path.join(output_dir, 'korean_lecture_features.pkl'), 'wb') as f:
        pickle.dump(final_data, f)


def process_text_file(args):
    """
    텍스트 처리용 단일 파일 함수 (멀티프로세싱용)
    """
    json_file, training_dir = args

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text = data["06_transcription"]["1_text"]
        major = data["03_lectureinfo"]["3_major_category"]
        file_id = data["01_dataset"]["1_identifier"]

        words = text.split()

        return {
            'words': words,
            'major': major,
            'text': text,
            'file_id': file_id,
            'status': 'success'
        }

    except Exception as e:
        return {'status': 'error', 'file': str(json_file), 'error': str(e)}


def get_rarewords_korean_mp(training_dir, output_dir, num_workers=None):
    """
    멀티프로세싱을 사용한 희귀 단어 추출
    training_dir: IT_Lecture/Training 경로
    """
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Using {num_workers} workers for text processing")

    os.makedirs(output_dir, exist_ok=True)

    # Label 폴더에서 JSON 파일 목록 수집
    label_dir = os.path.join(training_dir, "Label")
    json_files = list(Path(label_dir).glob("**/*.json"))
    print(f"Processing {len(json_files)} files for rare words")

    # 멀티프로세싱 인자 준비
    args_list = [(json_file, training_dir) for json_file in json_files]

    all_words = []
    major_words = {}
    utterance_data = []

    start_time = time.time()

    # 청크 단위로 처리
    chunk_size = 500

    for i in range(0, len(args_list), chunk_size):
        chunk = args_list[i:i + chunk_size]

        with Pool(num_workers) as pool:
            results = pool.map(process_text_file, chunk)

        # 결과 수집
        for result in results:
            if result['status'] == 'success':
                words = result['words']
                major = result['major']

                all_words.extend(words)

                if major not in major_words:
                    major_words[major] = []
                major_words[major].extend(words)

                utterance_data.append({
                    'text': result['text'],
                    'words': words,
                    'major': major,
                    'file_id': result['file_id']
                })

        elapsed = time.time() - start_time
        print(f"Text processed: {i + len(chunk)}/{len(json_files)} files ({elapsed:.1f}s)")

    # 나머지는 기존과 동일
    word_freq = Counter(all_words)
    rare_words = [word for word, freq in word_freq.items() if freq <= 3 and len(word) > 1]

    tech_terms = {
        'comp': ['알고리즘', '데이터베이스', '객체지향', '프로그래밍', '소프트웨어', '하드웨어',
                 '네트워크', '보안', '인공지능', '머신러닝', '딥러닝', '빅데이터'],
        'eng': ['미적분학', '선형대수', '열역학', '유체역학', '회로이론', '신호처리',
                '제어시스템', '구조역학', '재료공학', '기계설계'],
        'math': ['미분방정식', '확률론', '통계학', '집합론', '위상수학', '해석학',
                 '대수학', '기하학', '수치해석'],
        'phy': ['양자역학', '상대성이론', '전자기학', '고체물리', '원자물리', '핵물리']
    }

    major_rare_words = {}
    for major, words in major_words.items():
        major_freq = Counter(words)
        major_rare = [word for word, freq in major_freq.items() if freq <= 2 and len(word) > 1]
        major_rare_words[major] = major_rare

    final_rare_words = set(rare_words)

    for major, terms in tech_terms.items():
        final_rare_words.update(terms)
        if major in major_rare_words:
            final_rare_words.update(major_rare_words[major][:50])

    # 파일 저장
    with open(os.path.join(output_dir, 'korean_rareword_error.txt'), 'w', encoding='utf-8') as f:
        for word in sorted(final_rare_words):
            f.write(f"{word}\n")

    with open(os.path.join(output_dir, 'korean_all_rare_words.txt'), 'w', encoding='utf-8') as f:
        for word in sorted(final_rare_words):
            f.write(f"{word}\n")

    # 발화별 편향 단어 생성
    utterance_bias = {}
    for i, utt_data in enumerate(utterance_data):
        bias_words = [word for word in utt_data['words'] if word in final_rare_words]

        if bias_words:
            utterance_bias[f"utterance_{i}"] = {
                "text": utt_data['text'],
                "bias_words": bias_words,
                "major": utt_data['major'],
                "file_id": utt_data['file_id']
            }

    with open(os.path.join(output_dir, 'korean_lecture_bias.json'), 'w', encoding='utf-8') as f:
        json.dump(utterance_bias, f, ensure_ascii=False, indent=2)

    print(f"Total rare words: {len(final_rare_words)}")
    print(f"Utterances with bias words: {len(utterance_bias)}")
    print(f"Text processing completed in {time.time() - start_time:.1f}s")

    return final_rare_words, utterance_bias


def dump_feature_korean(training_dir, output_dir):
    """기존 함수 (호환성 유지)"""
    return dump_feature_korean_mp(training_dir, output_dir)


def get_rarewords_korean(training_dir, output_dir):
    """기존 함수 (호환성 유지)"""
    return get_rarewords_korean_mp(training_dir, output_dir)


# 사용 예시
if __name__ == "__main__":
    # 파일 구조에 맞는 경로 설정
    training_dir = "./IT_Lecture/Training"  # Training 폴더 경로
    output_dir = "./korean_processed"

    # CPU 코어 수에 따른 워커 수 설정
    num_workers = cpu_count()  # 최대 8개 프로세스
    print(f"Available CPU cores: {cpu_count()}, Using {num_workers} workers")

    # 경로 확인
    label_dir = os.path.join(training_dir, "Label")
    voice_dir = os.path.join(training_dir, "Voice")

    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found at {label_dir}")
        exit(1)
    if not os.path.exists(voice_dir):
        print(f"Error: Voice directory not found at {voice_dir}")
        exit(1)

    print(f"Label directory: {label_dir}")
    print(f"Voice directory: {voice_dir}")

    # 1. 특징 추출 (멀티프로세싱)
    print("\n=== Feature Extraction (Multiprocessing) ===")
    start_time = time.time()
    total_processed = dump_feature_korean_mp(training_dir, output_dir, num_workers)
    feature_time = time.time() - start_time
    print(f"Feature extraction completed: {total_processed} files in {feature_time:.1f}s")

    # 2. 희귀 단어 추출 (멀티프로세싱)
    print("\n=== Rare Words Extraction (Multiprocessing) ===")
    start_time = time.time()
    rare_words, bias_data = get_rarewords_korean_mp(training_dir, output_dir, num_workers)
    text_time = time.time() - start_time
    print(f"Text processing completed in {text_time:.1f}s")

    print("\n=== Processing Complete ===")
    print(f"Total time: {feature_time + text_time:.1f}s")
    print(f"Features saved to: {output_dir}/korean_lecture_features.pkl")
    print(f"Rare words saved to: {output_dir}/korean_rareword_error.txt")
    print(f"Bias data saved to: {output_dir}/korean_lecture_bias.json")
    print(f"Speed improvement: ~{num_workers}x faster than single-threaded")