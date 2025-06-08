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


def dump_feature_korean_safe(training_dir, output_dir, num_workers=None):
    """
    길이 통일된 mel spectrogram 생성 버전
    """
    if num_workers is None:
        num_workers = 4

    # Whisper 모델을 한 번만 로드
    model = whisper.load_model("medium")

    # 길이 설정 (Whisper 기본값)
    TARGET_LENGTH = 3000  # 30초 * 100 frames/sec = 3000 frames

    # 경로 정보만 멀티프로세싱으로 수집
    label_dir = os.path.join(training_dir, "Label")
    json_files = list(Path(label_dir).glob("**/*.json"))

    args_list = [(json_file, training_dir) for json_file in json_files]

    # 경로 정보 수집 (멀티프로세싱)
    with Pool(4) as pool:
        file_info_list = pool.map(process_single_file_light, args_list)

    valid_files = [info for info in file_info_list if info['status'] == 'success']

    # 순차적으로 Whisper 처리 (메모리 절약)
    features = []
    texts = []
    majors = []

    for i, info in enumerate(valid_files):
        try:
            # 1. 오디오 로드
            audio = whisper.load_audio(info['voice_path'])

            # 2. 길이 통일 (패딩/트리밍)
            audio = pad_or_trim_audio(audio, target_length=30.0)  # 30초로 통일

            # 3. mel spectrogram 생성
            mel = whisper.log_mel_spectrogram(audio)

            # 4. mel spectrogram도 길이 확인/통일
            mel = pad_or_trim_mel(mel, target_frames=TARGET_LENGTH)

            features.append(mel)
            texts.append(info['text'])
            majors.append(info['major'])

            if i % 100 == 0:
                print(f"Processed {i}/{len(valid_files)} files - mel shape: {mel.shape}")

        except Exception as e:
            print(f"Error processing {info['voice_path']}: {e}")
            continue

    # 저장 전 shape 확인
    if features:
        print(f"Final mel shapes: {[f.shape for f in features[:5]]}...")  # 처음 5개만 확인

    # 저장
    save_data = {'features': features, 'texts': texts, 'majors': majors}
    with open(os.path.join(output_dir, 'korean_lecture_features.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    return len(features)


def pad_or_trim_audio(audio, target_length=30.0):
    """
    오디오를 target_length 초로 패딩하거나 트리밍
    """
    target_samples = int(target_length * whisper.audio.SAMPLE_RATE)  # 16000 * 30 = 480000

    if len(audio) > target_samples:
        # 트리밍: 처음 30초만 사용
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        # 패딩: 0으로 채움
        padding_needed = target_samples - len(audio)
        audio = np.pad(audio, (0, padding_needed), mode='constant', constant_values=0)

    return audio


def pad_or_trim_mel(mel, target_frames=3000):
    """
    mel spectrogram을 target_frames로 패딩하거나 트리밍
    mel shape: (80, time_frames)
    """
    current_frames = mel.shape[1]

    if current_frames > target_frames:
        # 트리밍
        mel = mel[:, :target_frames]
    elif current_frames < target_frames:
        # 패딩
        padding_needed = target_frames - current_frames
        mel = np.pad(mel, ((0, 0), (0, padding_needed)), mode='constant', constant_values=mel.min())

    return mel


def validate_mel_shapes(features):
    """
    생성된 mel spectrogram들의 shape이 모두 동일한지 확인
    """
    if not features:
        return False, "No features"

    target_shape = features[0].shape
    for i, feat in enumerate(features):
        if feat.shape != target_shape:
            return False, f"Shape mismatch at index {i}: {feat.shape} vs {target_shape}"

    return True, f"All {len(features)} features have shape {target_shape}"


def dump_feature_korean_batch_safe(training_dir, output_dir, num_workers=None, chunk_size=1000):
    """
    대용량 데이터셋을 위한 청크별 처리 버전
    """
    if num_workers is None:
        num_workers = 4

    model = whisper.load_model("medium")
    TARGET_LENGTH = 3000

    # 경로 정보 수집
    label_dir = os.path.join(training_dir, "Label")
    json_files = list(Path(label_dir).glob("**/*.json"))
    args_list = [(json_file, training_dir) for json_file in json_files]

    with Pool(4) as pool:
        file_info_list = pool.map(process_single_file_light, args_list)

    valid_files = [info for info in file_info_list if info['status'] == 'success']

    # 청크별로 처리
    all_features = []
    all_texts = []
    all_majors = []

    for chunk_start in range(0, len(valid_files), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(valid_files))
        chunk_files = valid_files[chunk_start:chunk_end]

        print(f"Processing chunk {chunk_start // chunk_size + 1}: files {chunk_start}-{chunk_end}")

        chunk_features = []
        chunk_texts = []
        chunk_majors = []

        for i, info in enumerate(chunk_files):
            try:
                audio = whisper.load_audio(info['voice_path'])
                audio = pad_or_trim_audio(audio, target_length=30.0)
                mel = whisper.log_mel_spectrogram(audio)
                mel = pad_or_trim_mel(mel, target_frames=TARGET_LENGTH)

                chunk_features.append(mel)
                chunk_texts.append(info['text'])
                chunk_majors.append(info['major'])

            except Exception as e:
                print(f"Error processing {info['voice_path']}: {e}")
                continue

        # 청크 검증
        is_valid, msg = validate_mel_shapes(chunk_features)
        if not is_valid:
            print(f"Warning: {msg}")
        else:
            print(f"Chunk validation: {msg}")

        all_features.extend(chunk_features)
        all_texts.extend(chunk_texts)
        all_majors.extend(chunk_majors)

        print(f"Chunk {chunk_start // chunk_size + 1} completed: {len(chunk_features)} features")

    # 최종 검증
    is_valid, msg = validate_mel_shapes(all_features)
    print(f"Final validation: {msg}")

    # 저장
    save_data = {'features': all_features, 'texts': all_texts, 'majors': all_majors}
    with open(os.path.join(output_dir, 'korean_lecture_features.pkl'), 'wb') as f:
        pickle.dump(save_data, f)

    return len(all_features)


# 기존 함수들 (변경 없음)
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

        os.remove(chunk_file)

    final_data = {
        'features': all_features,
        'texts': all_texts,
        'majors': all_majors,
        'file_ids': all_file_ids
    }

    with open(os.path.join(output_dir, 'korean_lecture_features.pkl'), 'wb') as f:
        pickle.dump(final_data, f)


def process_text_file(args):
    """텍스트 처리용 단일 파일 함수 (멀티프로세싱용)"""
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
    """멀티프로세싱을 사용한 희귀 단어 추출"""
    if num_workers is None:
        num_workers = cpu_count()

    print(f"Using {num_workers} workers for text processing")

    os.makedirs(output_dir, exist_ok=True)

    label_dir = os.path.join(training_dir, "Label")
    json_files = list(Path(label_dir).glob("**/*.json"))
    print(f"Processing {len(json_files)} files for rare words")

    args_list = [(json_file, training_dir) for json_file in json_files]

    all_words = []
    major_words = {}
    utterance_data = []

    start_time = time.time()
    chunk_size = 500

    for i in range(0, len(args_list), chunk_size):
        chunk = args_list[i:i + chunk_size]

        with Pool(num_workers) as pool:
            results = pool.map(process_text_file, chunk)

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

    with open(os.path.join(output_dir, 'korean_rareword_error.txt'), 'w', encoding='utf-8') as f:
        for word in sorted(final_rare_words):
            f.write(f"{word}\n")

    with open(os.path.join(output_dir, 'korean_all_rare_words.txt'), 'w', encoding='utf-8') as f:
        for word in sorted(final_rare_words):
            f.write(f"{word}\n")

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
    return dump_feature_korean_safe(training_dir, output_dir)


def get_rarewords_korean(training_dir, output_dir):
    """기존 함수 (호환성 유지)"""
    return get_rarewords_korean_mp(training_dir, output_dir)


def process_single_file_light(args):
    """메모리 절약 버전 - Whisper 모델 로드 없이 경로만 확인"""
    json_file, training_dir = args

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        audio_relative_path = data["01_dataset"]["3_src_path"]
        text = data["06_transcription"]["1_text"]
        major = data["03_lectureinfo"]["3_major_category"]
        file_id = data["01_dataset"]["1_identifier"]

        audio_file_path = '/'.join(audio_relative_path.split('/')[-2:])
        voice_path = os.path.join(training_dir, "Voice", audio_file_path)

        if os.path.exists(voice_path):
            return {
                'voice_path': voice_path,
                'text': text,
                'major': major,
                'file_id': file_id,
                'status': 'success'
            }
        else:
            return {'status': 'audio_not_found', 'file': str(json_file)}

    except Exception as e:
        return {'status': 'error', 'file': str(json_file), 'error': str(e)}


# 사용 예시
if __name__ == "__main__":
    training_dir = "./IT_Lecture/Training"
    output_dir = "./korean_processed"

    num_workers = cpu_count()
    print(f"Available CPU cores: {cpu_count()}, Using {num_workers} workers")

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

    # 1. 특징 추출 (길이 통일된 버전)
    print("\n=== Feature Extraction (Length Normalized) ===")
    start_time = time.time()
    total_processed = dump_feature_korean_safe(training_dir, output_dir)
    feature_time = time.time() - start_time
    print(f"Feature extraction completed: {total_processed} files in {feature_time:.1f}s")

    # 2. 희귀 단어 추출
    print("\n=== Rare Words Extraction ===")
    start_time = time.time()
    rare_words, bias_data = get_rarewords_korean_mp(training_dir, output_dir, num_workers)
    text_time = time.time() - start_time
    print(f"Text processing completed in {text_time:.1f}s")

    print("\n=== Processing Complete ===")
    print(f"Total time: {feature_time + text_time:.1f}s")
    print(f"Features saved to: {output_dir}/korean_lecture_features.pkl")
    print(f"Note: All mel spectrograms are now normalized to (80, 3000) shape")