import os
from collections import defaultdict
from FrameClassifier import FrameClassifier


def evaluate_classifier_accuracy():
    classifier = FrameClassifier()
    results = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})

    base_path = 'datasets/dataset_test'
    manufacturers = ['altai', 'balakovo', 'begickaya', 'promlit',
                     'ruzhimmash', 'tvsz', 'uralvagon']

    for manufacturer in manufacturers:
        manufacturer_path = os.path.join(base_path, manufacturer)
        image_files = [f for f in os.listdir(manufacturer_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in image_files:
            img_path = os.path.join(manufacturer_path, img_name)
            results[manufacturer]['total'] += 1

            try:
                predicted = classifier.classify_frame(img_path)
                if predicted == manufacturer:
                    results[manufacturer]['correct'] += 1
                else:
                    results[manufacturer]['errors'].append((img_name, predicted))
            except Exception as e:
                results[manufacturer]['errors'].append((img_name, f"ERROR: {str(e)}"))

    for manufacturer in results:
        total = results[manufacturer]['total']
        correct = results[manufacturer]['correct']
        accuracy = correct / total * 100 if total > 0 else 0
        results[manufacturer]['accuracy'] = accuracy

    os.makedirs('results', exist_ok=True)
    save_results_to_file(results, 'results/classification_results.txt')
    print("\nРезультаты сохранены в файл 'classification_results.txt'")

    return results


def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        f.write("Результаты классификации по производителям\n")
        f.write("=" * 50 + "\n\n")

        for manufacturer, data in results.items():
            f.write(f"Производитель: {manufacturer}\n")
            f.write(f"Точность: {data['accuracy']:.2f}% ({data['correct']}/{data['total']})\n")

            if data['errors']:
                f.write("\nОшибки классификации:\n")
                for img_name, predicted in data['errors']:
                    f.write(f"  Изображение: {img_name}\n")
                    f.write(f"  Предсказанный класс: {predicted}\n")

            f.write("\n" + "-" * 50 + "\n")


if __name__ == '__main__':
    evaluation_results = evaluate_classifier_accuracy()

    # Дополнительный вывод в консоль
    print("\nИтоговая точность по производителям:")
    for manufacturer, data in evaluation_results.items():
        print(f"{manufacturer}: {data['accuracy']:.2f}%")