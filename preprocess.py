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


def process_batch_whisper_features(batch_args):
    """
    배치 단위로 Whisper 처리 (메모리 효율성 개선)
    한 워커가 여러 파일을 순차적으로 처리하여 모델 로딩 오버헤드 감소
    """
    batch_file_infos, model_name, target_length, all_rare_words = batch_args

    # 배치 처리용 모델 로드 (한 번만)
    model = whisper.load_model(model_name)
    target_frames = int(target_length * 100)  # 100 frames/sec

    results = []

    for i, file_info in enumerate(batch_file_infos):  # 👈 i 추가
        try:
            if i % 10 == 0:
                print(f"    배치 내 진행: {i}/{len(batch_file_infos)} 파일 처리 중...")

            # 오디오 처리
            audio = whisper.load_audio(file_info['voice_path'])
            audio = pad_or_trim_audio(audio, target_length=target_length)

            # mel spectrogram 생성
            mel = whisper.log_mel_spectrogram(audio)
            mel = pad_or_trim_mel(mel, target_frames=target_frames)

            # 바이어싱 단어
            utterance_words = set(file_info['text'].split())
            bias_words = list(utterance_words.intersection(all_rare_words))

            mel_numpy = mel.numpy() if isinstance(mel, torch.Tensor) else mel

            results.append({
                'status': 'success',
                'fbank': mel_numpy,
                'words': file_info['text'],
                'blist': bias_words,
                'voice_path': file_info['voice_path']
            })

        except Exception as e:
            results.append({
                'status': 'error',
                'error': str(e),
                'voice_path': file_info['voice_path']
            })

    return results


def process_single_whisper_feature(args):
    """
    단일 파일의 Whisper 특성 추출 (멀티프로세싱용)
    각 워커에서 Whisper 모델을 별도로 로드
    """
    file_info, model_name, target_length, all_rare_words = args

    try:
        # 각 워커에서 모델을 별도로 로드
        model = whisper.load_model(model_name)

        # 1. 오디오 로드
        audio = whisper.load_audio(file_info['voice_path'])

        # 2. 길이 통일 (패딩/트리밍)
        audio = pad_or_trim_audio(audio, target_length=target_length)

        # 3. mel spectrogram 생성
        mel = whisper.log_mel_spectrogram(audio)

        # 4. mel spectrogram 길이 통일
        target_frames = int(target_length * 100)  # 100 frames/sec
        mel = pad_or_trim_mel(mel, target_frames=target_frames)

        # 5. 바이어싱 단어 추출
        utterance_words = set(file_info['text'].split())
        bias_words = list(utterance_words.intersection(all_rare_words))

        # numpy로 변환 (pickle 호환성)
        mel_numpy = mel.numpy() if isinstance(mel, torch.Tensor) else mel

        return {
            'status': 'success',
            'fbank': mel_numpy,
            'words': file_info['text'],
            'blist': bias_words,
            'voice_path': file_info['voice_path']
        }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'voice_path': file_info['voice_path']
        }


def dump_feature_korean_tcpgen_format(training_dir, output_dir, num_workers=None, batch_size=50,
                                      use_batch_processing=True):
    """
    완전 멀티프로세싱 적용 TCPGEN WhisperBiasing 형식으로 mel spectrogram 생성
    출력: fbank.pt 파일 (키: fbank, words, blist)

    Args:
        training_dir: 훈련 데이터 디렉토리
        output_dir: 출력 디렉토리
        num_workers: 워커 수 (None이면 CPU 코어 수의 80%)
        batch_size: 배치 처리 시 배치 크기
        use_batch_processing: True면 배치 처리, False면 개별 처리
    """
    if num_workers is None:
        num_workers = max(1, int(cpu_count() * 0.8))  # CPU 코어의 80% 사용

    print(f"Using {num_workers} workers, batch_size={batch_size}, batch_processing={use_batch_processing}")

    # 경로 정보만 멀티프로세싱으로 수집
    label_dir = os.path.join(training_dir, "Label")
    json_files = list(Path(label_dir).glob("**/*.json"))

    args_list = [(json_file, training_dir) for json_file in json_files]

    # 경로 정보 수집 (멀티프로세싱)
    print("Collecting file information...")
    with Pool(num_workers) as pool:
        file_info_list = pool.map(process_single_file_light, args_list)

    valid_files = [info for info in file_info_list if info['status'] == 'success']
    print(f"Valid files: {len(valid_files)}")

    valid_files = valid_files[:len(valid_files) // 2]  # 앞쪽 50%만 사용

    # 바이어싱 단어 로드 (미리 생성된 희귀 단어 파일)
    rare_words_file = os.path.join(output_dir, 'korean_rareword_error.txt')
    if not os.path.exists(rare_words_file):
        print("Warning: 희귀 단어 파일이 없습니다. 먼저 get_rarewords_korean_mp() 실행이 필요합니다.")
        return 0

    with open(rare_words_file, 'r', encoding='utf-8') as f:
        all_rare_words = set(word.strip() for word in f.readlines())

    # Whisper 특성 추출 (완전 멀티프로세싱)
    model_name = "small"
    target_length = 30.0

    print(f"Starting Whisper feature extraction with {num_workers} workers...")
    start_time = time.time()

    if use_batch_processing:
        # 배치 처리 방식 (메모리 효율적)
        batches = [valid_files[i:i + batch_size] for i in range(0, len(valid_files), batch_size)]
        batch_args = [(batch, model_name, target_length, all_rare_words) for batch in batches]

        print(f"Processing {len(batches)} batches...")

        with Pool(num_workers) as pool:
            batch_results = pool.map(process_batch_whisper_features, batch_args)

        # 배치 결과 평면화
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)

    else:
        # 개별 처리 방식 (더 많은 병렬성)
        feature_args = [(info, model_name, target_length, all_rare_words) for info in valid_files]

        print(f"Processing {len(feature_args)} files individually...")

        with Pool(num_workers) as pool:
            all_results = pool.map(process_single_whisper_feature, feature_args)

    processing_time = time.time() - start_time
    print(f"Whisper processing completed in {processing_time:.1f}s")

    # 성공한 결과만 필터링
    successful_results = [r for r in all_results if r['status'] == 'success']
    failed_results = [r for r in all_results if r['status'] == 'error']

    print(f"Successful: {len(successful_results)}, Failed: {len(failed_results)}")

    if failed_results:
        print("Failed files (first 5):")
        for fail in failed_results[:5]:
            print(f"  {fail['voice_path']}: {fail['error']}")

    if not successful_results:
        print("No successful results to save!")
        return 0

    # TCPGEN 형식으로 저장
    fbank_features = [r['fbank'] for r in successful_results]
    words_list = [r['words'] for r in successful_results]
    blist_per_utterance = [r['blist'] for r in successful_results]

    # 저장 전 shape 확인
    if fbank_features:
        print(f"Final mel shapes: {[f.shape for f in fbank_features[:5]]}...")  # 처음 5개만 확인

    # TCPGEN 형식으로 저장
    tcpgen_data = {
        'fbank': fbank_features,  # mel spectrogram 리스트
        'words': words_list,  # 텍스트 리스트
        'blist': blist_per_utterance  # 각 발화별 바이어싱 단어 리스트
    }

    # .pt 파일로 저장 (TCPGEN이 torch.load()로 읽음)
    output_file = os.path.join(output_dir, 'fbank.pt')
    torch.save(tcpgen_data, output_file)
    print(f"TCPGEN format data saved to: {output_file}")

    # 추가로 .pkl 형식도 저장 (호환성)
    pickle_file = os.path.join(output_dir, 'korean_lecture_features_tcpgen.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(tcpgen_data, f)
    print(f"Pickle format data saved to: {pickle_file}")

    # 첫 번째 샘플 정보 출력
    if fbank_features:
        print(f"Sample mel shape: {fbank_features[0].shape}")
        print(f"Sample text: {words_list[0][:100]}...")
        print(f"Sample bias words: {blist_per_utterance[0][:5]}")

    return len(fbank_features)


def create_tcpgen_json_files(training_dir, output_dir, rare_words_file):
    """
    TCPGEN용 JSON 파일 생성
    - train_clean_100_error.json 형태
    - 각 발화별로 바이어싱 단어 포함
    """
    # 희귀 단어 로드
    with open(rare_words_file, 'r', encoding='utf-8') as f:
        rare_words = set(word.strip() for word in f.readlines())

    # 파일 정보 수집
    label_dir = os.path.join(training_dir, "Label")
    json_files = list(Path(label_dir).glob("**/*.json"))

    utterance_data = {}

    for i, json_file in enumerate(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text = data["06_transcription"]["1_text"]
            file_id = data["01_dataset"]["1_identifier"]

            # 이 발화의 바이어싱 단어
            utterance_words = set(text.split())
            bias_words = list(utterance_words.intersection(rare_words))

            if bias_words:  # 바이어싱 단어가 있는 경우만 포함
                utterance_data[f"utterance_{i}"] = {
                    "text": text,
                    "bias_words": bias_words,
                    "file_id": file_id,
                    "json_path": str(json_file)
                }

        except Exception as e:
            continue

    # JSON 파일로 저장
    json_output_file = os.path.join(output_dir, 'korean_train_error.json')
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(utterance_data, f, ensure_ascii=False, indent=2)

    print(f"TCPGEN JSON file created: {json_output_file}")
    print(f"Utterances with bias words: {len(utterance_data)}")

    return utterance_data


def validate_tcpgen_format(data_file):
    """
    TCPGEN 형식 데이터 검증
    """
    try:
        if data_file.endswith('.pt'):
            data = torch.load(data_file)
        else:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)

        required_keys = ['fbank', 'words', 'blist']
        for key in required_keys:
            if key not in data:
                return False, f"Missing key: {key}"

        fbank = data['fbank']
        words = data['words']
        blist = data['blist']

        if len(fbank) != len(words) or len(words) != len(blist):
            return False, f"Length mismatch: fbank={len(fbank)}, words={len(words)}, blist={len(blist)}"

        # shape 확인
        if fbank:
            sample_shape = fbank[0].shape
            for i, feat in enumerate(fbank[:10]):  # 처음 10개만 확인
                if feat.shape != sample_shape:
                    return False, f"Shape mismatch at index {i}: {feat.shape} vs {sample_shape}"

        return True, f"Valid TCPGEN format: {len(fbank)} utterances, feature shape: {fbank[0].shape if fbank else 'N/A'}"

    except Exception as e:
        return False, f"Error loading file: {e}"


def get_rarewords_korean_mp(training_dir, output_dir, num_workers=None):
    """
    멀티프로세싱을 사용한 희귀 단어 추출 (TCPGEN용)
    """
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

    # 한국어 기술 용어 추가
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

    # TCPGEN 형식 파일 저장
    with open(os.path.join(output_dir, 'korean_rareword_error.txt'), 'w', encoding='utf-8') as f:
        for word in sorted(final_rare_words):
            f.write(f"{word}\n")

    with open(os.path.join(output_dir, 'korean_all_rare_words.txt'), 'w', encoding='utf-8') as f:
        for word in sorted(final_rare_words):
            f.write(f"{word}\n")

    # 발화별 편향 단어 생성 (TCPGEN JSON 형식)
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

    with open(os.path.join(output_dir, 'korean_train_error.json'), 'w', encoding='utf-8') as f:
        json.dump(utterance_bias, f, ensure_ascii=False, indent=2)

    print(f"Total rare words: {len(final_rare_words)}")
    print(f"Utterances with bias words: {len(utterance_bias)}")
    print(f"Text processing completed in {time.time() - start_time:.1f}s")

    return final_rare_words, utterance_bias


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


# TCPGEN용 전체 파이프라인
def run_tcpgen_preprocessing_pipeline(training_dir, output_dir, num_workers=None, batch_size=50,
                                      use_batch_processing=True):
    """
    완전 멀티프로세싱 적용 TCPGEN WhisperBiasing용 전체 전처리 파이프라인
    1. 희귀 단어 추출
    2. TCPGEN 형식 특징 추출 (멀티프로세싱)
    3. JSON 파일 생성
    4. 검증

    Args:
        training_dir: 훈련 데이터 디렉토리
        output_dir: 출력 디렉토리
        num_workers: 워커 수 (None이면 CPU 코어 수의 80%)
        batch_size: 배치 처리 시 배치 크기
        use_batch_processing: True면 배치 처리, False면 개별 처리
    """

    print("=" * 60)
    print("TCPGEN WHISPERBIASING 전처리 파이프라인 (완전 멀티프로세싱)")
    print("=" * 60)

    if num_workers is None:
        num_workers = max(1, int(cpu_count() * 0.8))

    print(f"Using {num_workers} workers (CPU cores: {cpu_count()})")
    print(f"Batch processing: {use_batch_processing}, Batch size: {batch_size}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. 희귀 단어 추출
    print("\n1. 희귀 단어 추출 중...")
    start_time = time.time()
    rare_words, bias_data = get_rarewords_korean_mp(training_dir, output_dir, num_workers)
    text_time = time.time() - start_time
    print(f"희귀 단어 추출 완료: {len(rare_words)}개 단어, {text_time:.1f}초")

    # 2. TCPGEN 형식 특징 추출 (완전 멀티프로세싱)
    print("\n2. TCPGEN 형식 특징 추출 중 (멀티프로세싱)...")
    start_time = time.time()
    total_processed = dump_feature_korean_tcpgen_format(
        training_dir, output_dir, num_workers, batch_size, use_batch_processing
    )
    feature_time = time.time() - start_time
    print(f"특징 추출 완료: {total_processed}개 파일, {feature_time:.1f}초")

    if total_processed == 0:
        print("특징 추출 실패! 파이프라인을 중단합니다.")
        return 0

    # 3. 검증
    print("\n3. TCPGEN 형식 검증 중...")
    fbank_file = os.path.join(output_dir, 'fbank.pt')
    is_valid, msg = validate_tcpgen_format(fbank_file)
    print(f"검증 결과: {'✅ 성공' if is_valid else '❌ 실패'}")
    print(f"세부사항: {msg}")

    # 4. 결과 요약
    print("\n" + "=" * 60)
    print("전처리 완료!")
    print("=" * 60)
    print(f"총 처리 시간: {feature_time + text_time:.1f}초")
    print(f"Whisper 처리 시간: {feature_time:.1f}초 (멀티프로세싱 적용)")
    print(f"텍스트 처리 시간: {text_time:.1f}초")
    print(f"속도 향상: Whisper 처리가 {num_workers}개 워커로 병렬화됨")
    print(f"생성된 파일들:")
    print(f"  - fbank.pt (TCPGEN 메인 데이터)")
    print(f"  - korean_rareword_error.txt (바이어싱 단어 목록)")
    print(f"  - korean_train_error.json (발화별 바이어싱 단어)")
    print(f"  - korean_lecture_features_tcpgen.pkl (호환성용)")

    return total_processed


# 사용 예시
if __name__ == "__main__":
    training_dir = "./IT_Lecture/Training"
    output_dir = "./korean_processed_tcpgen"

    num_workers = max(1, int(cpu_count() * 0.8))  # CPU 코어의 80% 사용
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

    # TCPGEN 전처리 파이프라인 실행 (완전 멀티프로세싱)
    total_processed = run_tcpgen_preprocessing_pipeline(
        training_dir,
        output_dir,
        num_workers=num_workers,
        batch_size=30,  # 메모리에 따라 조절
        use_batch_processing=True  # 메모리 효율적
    )

    print(f"\n🎉 TCPGEN 전처리 완료!")
    print(f"이제 WhisperBiasing 레포의 train.py를 실행할 수 있습니다.")

    # 개별 처리 방식으로 실행하고 싶다면:
    # total_processed = run_tcpgen_preprocessing_pipeline(
    #     training_dir,
    #     output_dir,
    #     num_workers=4,  # 메모리 사용량이 높으므로 워커 수 줄임
    #     use_batch_processing=False
    # )