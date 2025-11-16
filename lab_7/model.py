import os
import re
import math
import chardet
from collections import Counter

class MailClassifierModel:
    # Ham болон spam файлуудыг агуулсан фолдерийн замыг байгуулагч руу дамжуулж моделыг сургана
    def __init__(self, folder_path_ham, folder_path_spam):
        print("📘 Сургаж байна...")
        self.folder_path_ham = folder_path_ham
        self.folder_path_spam = folder_path_spam

        # Ham, spam тус бүрд нь unigram-ыг үүсгэнэ (Counter объект буцаана)
        self.ham_unigram = self.createUnigram(folder_path_ham)
        self.spam_unigram = self.createUnigram(folder_path_spam)

        # Unigram-уудыг текст файл дотор хадгална
        self.saveUnigram(self.ham_unigram, "ham_unigram.txt")
        self.saveUnigram(self.spam_unigram, "spam_unigram.txt")

        # Моделыг сургана
        self.data = {}
        self.train()

        # Үр дүнг хэвлэнэ
        self.printResult()

    # Файлыг уншууд текстыг цэвэрлэнэ
    def getCleanWords(self, file_path):
        try:
            # Файлыг байтаар нь уншина
            with open(file_path, "rb") as f:
                raw_data = f.read()
            # Encoding-ыг нь тодорхойлно
            result = chardet.detect(raw_data)
            encoding = result["encoding"] or "utf-8"
            # Тодорхойлсон encoding-г ашиглаж decode хийнэ
            text = raw_data.decode(encoding, errors="ignore")

            # Regular expression ашиглаж текстыг цэвэрлэнэ
            cleaned = re.sub(r'[^a-zA-Z가-힣\s]', '', text).lower()
            return cleaned
        
        # Файл уншихад алдаа гарвал мэдэгдэнэ
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")
            return ""

    # Файлын замыг дамжуулж өгөөд уншаад Unigram-г үүсгэнэ
    def createUnigram(self, folder_path):
        # Counter объект (dictionary) ашиглаж үгний тоог хадгална
        unigram = Counter()

        # Тухайн фолдер доторх файлын тоогоор давтана
        for filename in os.listdir(folder_path):
            # Файл нь текст биш өргөтгөлтэй бол алгасна
            if not filename.endswith(".txt"):
                continue
            # Файлын нэрийг бүтээнэ
            file_path = os.path.join(folder_path, filename)
            # Цэвэрлэсэн текстыг авна
            cleaned = self.getCleanWords(file_path)
            # Текст доторх үгнүүдийг салгана
            words = cleaned.split()
            # Unigram руу оруулна
            unigram.update(words)
        return unigram

    # Unigram-г текст файл руу хадгална
    def saveUnigram(self, unigram, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for word, count in unigram.most_common():
                f.write(f"{word} {count}\n")
        print(f"✅ {filename}-г хадгаллаа")

    # Моделыг сургана
    def train(self):
        # Ялгаатай үгийн олонлог
        vocab = set(self.ham_unigram.keys()) | set(self.spam_unigram.keys())
        # Ялгаатай үгийн тоо
        vocab_size = len(vocab)

        # Unigram тус бүрийн нийт үгийн тоо
        total_ham_words = sum(self.ham_unigram.values())
        total_spam_words = sum(self.spam_unigram.values())

        # Ham болон spam файлын тоо
        n_spam_files = len(os.listdir(self.folder_path_spam))
        n_ham_files = len(os.listdir(self.folder_path_ham))
        total_files = n_spam_files + n_ham_files

        # Ham болон spam-н prior магадлал
        p_spam = n_spam_files / total_files
        p_ham = n_ham_files / total_files

        # Дээрх тооцооллуудыг шинжийн утгад хадгална
        self.data = {
            "vocab": vocab,
            "vocab_size": vocab_size,
            "ham_total": total_ham_words,
            "spam_total": total_spam_words,
            "ham_counts": self.ham_unigram,
            "spam_counts": self.spam_unigram,
            "p_spam": p_spam,
            "p_ham": p_ham,
        }

        print("✅ Сургалт амжилттай.")

    # 1 файлын таамаглалыг гаргана
    def predict(self, file_path):
        # Файл доторх үгийн жагсаалтыг гаргана
        cleaned = self.getCleanWords(file_path)
        words = cleaned.split()

        # Үг байхгүй бол алдаа заана
        if not words:
            print("⚠️ Файлыг уншихад алдаа гарлаа!")
            return "unknown"
        # Laplace smoothing
        a = 1
        # Prior магадлалын логарифм
        log_prob_ham = math.log(self.data["p_ham"])
        log_prob_spam = math.log(self.data["p_spam"])

        # Үг болгоны магадлалын логарифмыг тооцож нэмнэ
        for w in words:
            p_w_ham = (self.data["ham_counts"][w] + a) / (
                self.data["ham_total"] + a * self.data["vocab_size"]
            )
            p_w_spam = (self.data["spam_counts"][w] + a) / (
                self.data["spam_total"] + a * self.data["vocab_size"]
            )
            log_prob_ham += math.log(p_w_ham)
            log_prob_spam += math.log(p_w_spam)

        # Аль магадлал нь их байгаагаар ангилагдана
        return "spam" if log_prob_spam > log_prob_ham else "ham"

    # Бүтэн тест явуулна
    def evaluate(self, dev_folder_path, verbose=True):
        correct = 0
        total = 0

        correct_ham = 0
        correct_spam = 0
        total_ham = 0
        total_spam = 0

        # Ham өгөгдөл дээр туршина
        ham_folder = os.path.join(dev_folder_path, "ham")
        for filename in os.listdir(ham_folder):
            if not filename.endswith(".txt"):
                continue
            file_path = os.path.join(ham_folder, filename)
            prediction = self.predict(file_path)
            if prediction == "ham":
                correct += 1
                correct_ham += 1
            total_ham += 1
            total += 1
            # Дэлгэрэнгүй хэвлэнэ
            # if verbose:
            #     print(f"[HAM] {filename}: {prediction}")

        # Spam өгөгдөл дээр туршина
        spam_folder = os.path.join(dev_folder_path, "spam")
        for filename in os.listdir(spam_folder):
            if not filename.endswith(".txt"):
                continue
            file_path = os.path.join(spam_folder, filename)
            prediction = self.predict(file_path)
            if prediction == "spam":
                correct += 1
                correct_spam += 1
            total_spam += 1
            total += 1
            # Дэлгэрэнгүй хэвлэнэ
            # if verbose:
            #     print(f"[SPAM] {filename}: {prediction}")

        # Хэр оновчтой хариулсныг тооцоолно
        accuracy = correct / total if total > 0 else 0
        print(f"\n🎯 Accuracy: {accuracy * 100:.2f}% "
              f"({correct}/{total})")
        error_rate = (total - correct) / total if total > 0 else 0
        print(f"🎯 Error rate: {error_rate * 100:.2f}% "
              f"({total - correct}/{total})")

        print(f"\nConfusion matrix:")
        print(f"                    Classified as:  ")
        print(f"                 |    0   |    1    ")
        print(f"Correct label: 0 |  {correct_spam}   |   {total_spam - correct_spam}")
        print(f"Correct label: 1 |   {total_ham - correct_ham}   |  {correct_ham}")

        precision = correct_ham / (correct_ham + total_spam - correct_spam)
        recall = correct_ham / total_ham
        f1 = 2* (precision*recall)/(precision+recall)

        print(f"False alarm rate: {(total_spam - correct_spam)*100/total_spam:.2f}%")
        print(f"Missed detection rate: {(total_ham - correct_ham)*100/total_ham:.2f}%")
        print(f"Precision: { precision* 100:.2f}%")
        print(f"Recall: { recall*100:.2f}%")
        print(f"F1 score: {f1*100:.2f}%")


        return accuracy
    

    # Үр дүнг хэвлэнэ
    def printResult(self):
        print(f"Моделын сургалтын үр дүн:")
        print(f"📚 Ялгаатай үгийн тоо: {self.data['vocab_size']}")
        print(f"📨 Нийт ham үгийн тоо: {self.data['ham_total']}")
        print(f"📨 Нийт spam үгийн тоо: {self.data['spam_total']}")
        print(f"⚖️  P(ham): {self.data['p_ham']:.3f}, P(spam): {self.data['p_spam']:.3f}")

    # Файл болгон дахь үгийн тоог хэвлэнэ
    def saveDetailed(self, folder_path, output_filename):
        vocab = sorted(list(self.data["vocab"]))
        with open(output_filename, "w", encoding="utf-8") as f:
            # Толгой хэсэг
            f.write("filename," + ",".join(vocab) + "\n")

            # Файл болгоноор давтана
            for filename in os.listdir(folder_path):
                if not filename.endswith(".txt"):
                    continue
                file_path = os.path.join(folder_path, filename)
                words = self.getCleanWords(file_path).split()
                counter = Counter(words)

                # Тухайн файл доторх үгийн тоог хэвлэнэ
                row_values = [str(counter.get(word, 0)) for word in vocab]
                f.write(filename + "," + ",".join(row_values) + "\n")

        print(f"✅ '{output_filename}'-г хадгаллаа")


# Main
if __name__ == "__main__":
    folder_path_ham = "data/train/ham"
    folder_path_spam = "data/train/spam"
    model = MailClassifierModel(folder_path_ham, folder_path_spam)

    # # Ганц файл дээр таамаглал хийнэ
    # model.predict(r"data/dev/ham/0689.2000-03-22.farmer.ham.txt")
    # model.predict(r"data/dev/spam/0006.2003-12-18.GP.spam.txt")

    # Нийт файл дээр таамаглал хийнэ
    model.evaluate("data/dev")

    model.saveDetailed(folder_path_ham, "details_ham.txt")
    model.saveDetailed(folder_path_spam, "details_spam.txt")

    # precision ✅
    # accuracy  ✅
    # recall
    # confusion matrix
