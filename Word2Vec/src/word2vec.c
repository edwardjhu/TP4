//  Copyright 2021 Microsoft Corporation
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#include <time.h>
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 12
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40



const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char file_suffix[200];
struct vocab_word *vocab;
int binary = 0, cbow = 1, debug_mode = 0, window = 5, min_count = 5, num_threads = 12, min_reduce = 1, oh = 0;
int *vocab_hash;
long long vocab_max_size = 1000000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, discard_word_count_actual = 0, iter = 5, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real beta = 0.9999, wd = 0.0, _alpha_wd = 0.0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

real loss = 0.0;

int which_thread_to_print = 0;
int which_thread_to_save = 0;
int num_thread_finished = 0;
int active_threads = 0;

pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock_emb0 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock_emb1 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

char *my_itoa(long long num, char *str) {
  if(str == NULL) return NULL;
  sprintf(str, "%lld", num);
  return str;
}

void prepend(char* s, const char* t) {
  size_t len = strlen(t);
  memmove(s + len, s, strlen(s) + 1);
  memcpy(s, t, len);
}

void *SaveCheckpoint(void *suffix) {
  long a, b, c, d;
  FILE *fo;
  char file_name[100];
  strcpy(file_name, output_file);
  strcat(file_name, (char *)suffix);
  fo = fopen(file_name, "wb");
  // Save the word vectors
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    fprintf(fo, "%s ", vocab[a].word);
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}


void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;

#ifdef _MSC_VER
  syn0 = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#elif defined  linux
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif

  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
#ifdef _MSC_VER
	  syn1 = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#elif defined  linux
	  a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif    
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
#ifdef _MSC_VER
	  syn1neg = _aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
#elif defined  linux
	  a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
#endif
    
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
    if (oh > 0) {
      if ((a+vocab_size)==b) {
        syn1neg[a * layer1_size + b] = 0;
      } else {
        syn1neg[a * layer1_size + b] = 0;
      }
    } else syn1neg[a * layer1_size + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
    if (oh > 0) {
      if (a==b) {
        syn0[a * layer1_size + b] = 1;
      } else {
        syn0[a * layer1_size + b] = 0;
      }
    } else syn0[a * layer1_size + b] = ((rand() / (real)RAND_MAX) - 0.5) * 2 * sqrt(3) / sqrt(layer1_size); // random init
  }
  CreateBinaryTree();
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, discard_word_count = 0, last_word_count = 0, last_discard_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g, prob;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  real acc_loss = 0;
  FILE *fi = fopen(train_file, "rb");
  
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count >= 500000 / num_threads) {
      pthread_mutex_lock(&lock);
      num_thread_finished++;
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      pthread_mutex_unlock(&lock);
      if ((long long)id == 0) { // syncing
        while (num_thread_finished < active_threads) {
          sleep(1);
        }
        pthread_mutex_lock(&lock);
        num_thread_finished = 0;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&lock);
        sleep(1);
      } else {
        pthread_mutex_lock(&lock);
        pthread_cond_wait(&cond, &lock);
        pthread_mutex_unlock(&lock);
      }

      pthread_mutex_lock(&lock);
      if ((debug_mode > -1) && ((long long)id == 0)) {
        printf("%f\t%.2f\t%lld\t%.10f\n", alpha,
         (word_count_actual) / (real)(iter * train_words + 1) * 100,
         word_count_actual,
         loss);
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - (word_count_actual) / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      _alpha_wd = 1 - alpha * wd;
      pthread_mutex_unlock(&lock);
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          if (ran < rand() / (real)RAND_MAX) {
            continue;
          }
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count >= train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      //syncing all threads before ckpting
      if ((long long)id == 0) {
        pthread_mutex_lock(&lock);
        file_suffix[0] = '\0';
        my_itoa(iter - local_iter, file_suffix);
        prepend(file_suffix, ".iter");
        SaveCheckpoint((void *)file_suffix);
        pthread_mutex_unlock(&lock);
      }
      
      if ((local_iter == 0) & ((long long)id == 0)) exit(0);
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    b = rand() % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      pthread_mutex_lock(&lock_emb0);
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      pthread_mutex_unlock(&lock_emb0);
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        acc_loss = 0;
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            target = table[rand() % table_size];
            if (target == 0) target = rand() % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          pthread_mutex_lock(&lock_emb1);
          for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
          pthread_mutex_unlock(&lock_emb1);
          if (f > MAX_EXP) prob = 1;
          else if (f < -MAX_EXP) prob = 0;
          else prob = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          g = (label - prob) * alpha;
          // log loss
          real this_loss;
          if (label == 0) this_loss = -log(1-prob+1e-10);
          else this_loss = -log(prob+1e-10);
          acc_loss += this_loss;

          pthread_mutex_lock(&lock_emb1);
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) {
            syn1neg[c + l2] *= _alpha_wd;
            syn1neg[c + l2] += g * neu1[c];
          }
          pthread_mutex_unlock(&lock_emb1);
        }
        acc_loss = (1-beta)*acc_loss / negative;
        pthread_mutex_lock(&lock);
        loss *= beta; 
        loss += acc_loss;
        pthread_mutex_unlock(&lock);
        // hidden -> in
        pthread_mutex_lock(&lock_emb0);
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < layer1_size; c++) {
            syn0[c + last_word * layer1_size] *= _alpha_wd;
            syn0[c + last_word * layer1_size] += neu1e[c] / cw;
          }
        }
        pthread_mutex_unlock(&lock_emb0);
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  pthread_mutex_lock(&lock);
  active_threads--;
  pthread_mutex_unlock(&lock);
  fclose(fi);
  free(neu1);
  free(neu1e);
#ifdef _MSC_VER
_endthreadex(0);
#elif defined  linux 
  pthread_exit(NULL);
#endif
}

#ifdef _MSC_VER
DWORD WINAPI TrainModelThread_win(LPVOID tid){
	TrainModelThread(tid);
	return 0;
}
#endif

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  if (debug_mode > 0) printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  
#ifdef _MSC_VER
	HANDLE *pt = (HANDLE *)malloc(num_threads * sizeof(HANDLE));
	for (int i = 0; i < num_threads; i++){
		pt[i] = (HANDLE)_beginthreadex(NULL, 0, TrainModelThread_win, (void *)i, 0, NULL);
	}
	WaitForMultipleObjects(num_threads, pt, TRUE, INFINITE);
	for (int i = 0; i < num_threads; i++){
		CloseHandle(pt[i]);
	}
	free(pt);
#elif defined  linux 
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  active_threads = num_threads;
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
#endif
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  cbow = 1;
  alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-beta", argc, argv)) > 0) beta = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-wd", argc, argv)) > 0) wd = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  hs = 0;
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-oh", argc, argv)) > 0) oh = atoi(argv[i + 1]);
  
  printf("ver 0.0.13 wd: %f, alpha: %f\n", wd, alpha);
  fflush(stdout);

  _alpha_wd = 1 - alpha * wd;
  
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}