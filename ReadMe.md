# ReadMe

## Sposób użycia

- Zainstaluj pythona w wersji od 3.8 do 3.11, aby jego wersja była kompatybilna ze wszystkimi bibliotekami
- Pobierz biblioteki z requirements.txt

- Wybierz plik

- **data_visualisation.ipynb**: Wizualizacja zbioru danych i sprawdzenie działania różnych filtrów
- **get_rectangle.ipynb**: Samodzielne trenowanie modelu do detekcji ręki. 
- **machine_learning.ipynb**: Wyniki zastosowania modelu uczenia maszynowego
- **deep_learning.ipynb**: Wyniki zastosowania modelu uczenia głębokiego
- **compare_models.ipynb**: Porównanie czasu działania modeli
- **main.py** - Testowanie modelów w czasie rzeczywistym

- **game_menu.py** - Wyświetlenie rankingu najlepszych graczy, możliwośc wpisania własnego imienia oraz pokazanie się okna gry.
Należy w terminal wpisać komendę: 

streamlit run game_menu.py


## Przegląd Projektu

Celem tego projektu jest stworzenie modelu, który będzie wykrywał dłoń i rozpoznawał pokazywaną liczbę palców, zarówno na zdjęciach, jak i w czasie rzeczywistym przy użyciu kamery internetowej. Projekt porównuje dwa podejścia: jedno korzystające z uczenia głębokiego (deep learning) z obrazami dłoni, a drugie z uczeniem maszynowym (machine learning) przy użyciu punktów charakterystycznych dłoni (współrzędne punktów palców, kąty między palcami). Celem jest zidentyfikowanie najlepszego podejścia do wykrywania dłoni i zliczania palców poprzez porównanie różnych modeli i technik.

### Struktura Folderów

- **train/**: Zawiera zdjęcia treningowe i adnotacje.
- **test/**: Zawiera zdjęcia testowe i adnotacje.
- **venv/**: Wirtualne środowisko dla projektu.
- **.idea/**: Pliki konfiguracyjne dla IDE.

### Kluczowe Pliki i Moduły

- **A-gray-model.pt, A-HoG-model.pt, A-LoG-model.pt, Simple-gray-model.pt, Simple-HoG-model.pt, Simple-LoG-model.pt**: Wytrenowane modele CNN używane w różnych modułach, zapisane
przy pomocy pytorch.save
- **SVC()-model.joblib, LinearDiscriminantAnalysis()-model.joblib, DecisionTreeClassifier()-model.joblib**: Wytrenowane modele uczenia maszynowego zapisane za pomocą joblib.
- **camera_prediction.py, camera_snake.py**: Moduły, które demonstrują praktyczne zastosowanie zliczania palców przy użyciu kamery internetowej.
- **compare_models.ipynb**: Wizualizacja i porównanie szybkości oraz wydajności modeli CNN i SVM.
- **count_fingers_cnn.py**: Funkcje związane z modelami CNN, w tym trenowanie i wizualizacja wyników. Te funkcje są używane w innych modułach.
- **data_func.py**: Funkcje używane do manipulacji danymi.
- **data_visualisation.ipynb**: Wizualizuje dane, pokazuje efekt różnych filtrów i sprawdza, czy klasy są zbalansowane.
- **deep_learning.ipynb**: Przetwarza dane dla CNN, wizualizuje proces uczenia się i porównuje różne CNN oraz filtry na danych.
- **detect_hand_cnn.py**: Używa CNN do wykrywania kluczowych punktów w celu wycięcia dłoni ze zdjęć.
- **game_menu.py** - Wyświetlenie rankingu najlepszych graczy, możliwośc wpisania własnego imienia oraz pokazanie się okna gry.
- **get_rectangle.ipynb**: Pokazuje, jak model nauczył się wycinać dłonie ze zdjęć.
- **hand_func.py**: Funkcje używające MediaPipe do znajdywania landmarków dłoni.
- **main.py**: Wywołanie wskazanych funkcji w czasie rzeczywistym.
- **machine_learning.ipynb**: Znajduje cechy, przetwarza dane, determinuje najlepszy algorytm i hiperparametry oraz wizualizuje wyniki.
- **model_prediction.py**: Funkcje umożliwiające predykowanie wyników przy użyciu istniejących modeli i wizualizujące ich wydajność.
- **scores.pkl**: Plik zapisujący najlepszych graczy.
- **tensor_data.py**: Dostosowuje format danych do tensorów i dataloaderów, aby umożliwić efektywne trenowanie modelu.
- **data.csv, data_normalized.csv, selected_features.csv**, **output.xlsx**: Plik wynikowy zawierający wstępnie zapisane dane.

## Cele Projektu

- **Wykrywanie Dłoni**: Opracowanie systemu wykrywającego i wycinającego dłonie ze zdjęć.
- **Zliczanie Palców**: Rozpoznawanie liczby pokazywanych palców na zdjęciach lub w czasie rzeczywistym.
- **Porównanie Modeli**: Porównanie wydajności modeli CNN i SVM w zadaniu.
- **Inżynieria Cech**: Identyfikacja i wykorzystanie odpowiednich cech, takich jak współrzędne punktów palców i kąty między palcami, dla modeli uczenia maszynowego.
- **Wizualizacja Danych**: Użycie różnych technik wizualizacji do zrozumienia danych i wydajności modeli.
- **Zastosowanie w Czasie Rzeczywistym**: Dostosowanie modeli do zliczania palców w czasie rzeczywistym z akceptowalną wydajnością.

## Wyzwania Projektu

- **Dobór Cech**: Kluczowy dla modeli uczenia maszynowego w celu poprawy dokładności i zmniejszenia złożoności.
- **Reprezentacja Danych**: Istotna dla modeli uczenia głębokiego, aby zapewnić skuteczne uczenie się i wydajność.
- **Wydajność w Czasie Rzeczywistym**: Zbalansowanie dokładności modelu i jego złożoności w celu osiągnięcia szybkich predykcji w aplikacjach działających w czasie rzeczywistym.

## Technologie i Biblioteki

- **Widzenie Komputerowe**: Techniki wykrywania obiektów, wykrywanie tła i analiza gradientów.
- **Uczenie Głębokie**: CNN do detekcji dłoni i zliczania palców na podstawie obrazów.
- **Uczenie Maszynowe**: Modele oparte na cechach, takie jak SVM i Klasyfikator Drzewa Decyzyjnego.
- **MediaPipe**: Do wykrywania punktów charakterystycznych dłoni.
- **Wizualizacja**: Narzędzia do wizualizacji danych i wyników w celu analizy i porównania wydajności modeli.
