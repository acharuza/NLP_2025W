import json
import os
import glob

# Gdzie szukać plików (folder data/results oraz bieżący)
SEARCH_DIRS = ["data/results", "."]


def review_human_verified():
    # Znajdź wszystkie pliki z "FINAL" w nazwie
    found_files = []
    for d in SEARCH_DIRS:
        found_files.extend(glob.glob(os.path.join(d, "*results*.json")))

    # Usuń duplikaty i posortuj
    found_files = sorted(list(set(found_files)))

    if not found_files:
        print("❌ Nie znaleziono plików *FINAL*.json. Uruchom najpierw skrypty naprawcze (finalize_...).")
        return

    print(f"🔎 Przeszukiwanie {len(found_files)} plików pod kątem ręcznej weryfikacji...\n")

    total_verified = 0

    for filepath in found_files:
        filename = os.path.basename(filepath)

        # Wykrycie modelu dla czytelności
        model_label = "MODEL: NIEZNANY"
        if "llama" in filename.lower():
            model_label = "🟢 MODEL: LLAMA 3.1"
        elif "mistral" in filename.lower():
            model_label = "🔵 MODEL: MISTRAL"
        elif "llava" in filename.lower():
            model_label = "🟠 MODEL: LLaVA"

        print(f"\n{'=' * 80}")
        print(f"{model_label}  |  Plik: {filename}")
        print(f"{'=' * 80}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Obsługa formatu dict/list
            results = data['results'] if isinstance(data, dict) and 'results' in data else data

            count_in_file = 0

            for item in results:
                reasoning = item.get('evaluation_reasoning', '')

                # Szukamy znacznika HUMAN VERIFIED
                if "HUMAN VERIFIED" in reasoning:
                    count_in_file += 1
                    total_verified += 1

                    # Kolorowanie wyniku (tylko w terminalu obsługującym ANSI)
                    score_icon = "✅ PASS (1)" if item['evaluation_score'] == 1 else "❌ FAIL (0)"

                    print(f"🆔 ID: {item['id']}")
                    print(f"🏆 Wynik: {score_icon}")
                    print(f"💬 Prompt (fragment): \"{item['prompt'][:100]}...\"")
                    print(f"🤖 Odpowiedź (fragment): \"{item['model_response'][:150]}...\"")
                    print(f"👀 Uzasadnienie (Human): {reasoning}")
                    print("-" * 80)

            if count_in_file == 0:
                print("   (Brak ręcznych poprawek w tym pliku)")
            else:
                print(f"   ---> Znaleziono {count_in_file} poprawek w tym pliku.")

        except Exception as e:
            print(f"❌ Błąd odczytu pliku: {e}")

    print("\n" + "=" * 80)
    print(f"🏁 PODSUMOWANIE: Łącznie znaleziono {total_verified} przypadków zweryfikowanych przez człowieka.")
    print("=" * 80)


if __name__ == "__main__":
    review_human_verified()