document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let currentMode = 10; // Default to 10 cards
    let selectedCards = []; // Array of card objects {cardData, isReversed, element}
    let deck = [...TAROT_DATA]; // Copy of data

    // --- DOM Elements ---
    const gridEl = document.getElementById('grid');
    const btnShuffle = document.getElementById('btn-shuffle');
    const modeButtons = document.querySelectorAll('.btn-mode');
    const cardCountEl = document.getElementById('card-count');
    const toggleNamesEl = document.getElementById('toggle-names');
    const btnTheme = document.getElementById('btn-theme');

    // --- Initialization ---
    init();

    function init() {
        renderGrid();
        updateStatus();

        // Event Listeners
        btnShuffle.addEventListener('click', shuffleDeck);

        modeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                setMode(parseInt(e.target.dataset.mode));
            });
        });

        toggleNamesEl.addEventListener('change', (e) => {
            const names = document.querySelectorAll('.card-name');
            names.forEach(el => {
                if (e.target.checked) el.classList.remove('hidden-name');
                else el.classList.add('hidden-name');
            });
        });

        btnTheme.addEventListener('click', () => {
            document.body.classList.toggle('light-theme');
            btnTheme.textContent = document.body.classList.contains('light-theme') ? '☀️' : '🌙';
        });

        // Prompt Logic
        document.getElementById('btn-generate').addEventListener('click', generatePrompt);
        document.getElementById('btn-copy').addEventListener('click', copyPrompt);
    }

    // --- Core Logic ---

    function setMode(mode) {
        currentMode = mode;
        // Reset selections when changing mode
        deselectAll();

        // Update UI
        modeButtons.forEach(btn => btn.classList.remove('active'));
        document.querySelector(`.btn-mode[data-mode="${mode}"]`).classList.add('active');

        updateStatus();
    }

    function updateStatus() {
        cardCountEl.textContent = `Seleccionadas: ${selectedCards.length} / ${currentMode}`;
    }

    function fisherYatesShuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    function shuffleDeck() {
        // Shuffle the deck array
        deck = fisherYatesShuffle([...TAROT_DATA]);

        // Re-render grid with animation feel
        // In a real DOM manipulation, re-rendering 78 divs is cheap enough
        deselectAll();
        renderGrid();
    }

    function renderGrid() {
        gridEl.innerHTML = '';

        deck.forEach(cardData => {
            const cardContainer = document.createElement('div');
            cardContainer.className = 'card-container';

            // Randomize reversed status for visuals
            // Let's decide: Is the card reversed *on the table* before picking? 
            // Yes, "Indicador visual de posición (normal/reversa)".

            const isReversed = Math.random() < 0.5;

            const card = document.createElement('div');
            card.className = `card ${isReversed ? 'reversed' : ''}`;

            // Front (Tarot Image) - In DOM terms this is the "Back" of the element if 3D flipped
            // But visually, the user sees the "Back of the card" (pattern) initially?
            // Wait, usually you pick from face down cards.
            // "Selección por clic individual... Resaltado visual"
            // "Representación visual: Imagen... Nombre... Indicador"
            // If they are face down, you can't see the image.
            // Assumption: Cards are Face Down (Pattern visible). Users pick them.
            // After picking, or perhaps for "Generation", they are revealed?
            // Re-reading: "Representación visual: Cada carta debe mostrar: Imagen... Nombre"
            // This might mean they are Face Up?
            // "Mezcla aleatoria... Asignación aleatoria de posición"
            // If they are Face Up from the start, "1, 3, 10, 13" selection implies picking specific ones.
            // Let's assume **Face Down** initially for a reading experience, revealing on specific UI action, 
            // OR Face Up if it's a study tool.
            // Given "Botón Generar Prompt", usually you pick Blindly.
            // BUT "Representación visual" requirement lists Image/Name. 
            // Let's go with: Cards are Face Down. Clicking Selects them. 

            // Refined Plan: 
            // Card has 3D flip.
            // Default: Face Down.
            // Click: Selects it (Highlight).
            // "Generar Prompt" reveals them? Or do they flip on specific interaction?

            // Also, seeing 78 cards is cool.

            // Middle Ground: Cards are Face Down (Back Pattern).
            // When selected, they flip Face Up to reveal what you got.

            card.innerHTML = `
                <div class="card-face card-back"></div>
                <div class="card-face card-front">
                    <img src="${cardData.img}" alt="${cardData.name_es}">
                    <div class="position-indicator text-xs absolute top-1 right-1 bg-black text-white px-1 rounded">${isReversed ? 'REV' : 'UP'}</div>
                </div>
            `;

            const nameEl = document.createElement('div');
            nameEl.className = 'card-name';
            if (!toggleNamesEl.checked) nameEl.classList.add('hidden-name');
            nameEl.textContent = cardData.name_es;

            cardContainer.appendChild(card);
            cardContainer.appendChild(nameEl);

            // Click Event
            cardContainer.addEventListener('click', () => {
                toggleSelection(cardContainer, card, cardData, isReversed);
            });

            // Double Click for Modal
            cardContainer.addEventListener('dblclick', (e) => {
                e.stopPropagation(); // Prevent selection toggle jitter if possible
                showModal(cardData);
            });

            gridEl.appendChild(cardContainer);
        });
    }

    function deselectAll() {
        selectedCards = [];
        document.querySelectorAll('.card.selected').forEach(el => {
            el.classList.remove('selected');
            el.classList.remove('flipped'); // Reset flip on new game
        });
        updateStatus();
    }

    function toggleSelection(container, cardEl, cardData, isReversed) {
        const isSelected = cardEl.classList.contains('selected');

        if (isSelected) {
            // Deselect
            cardEl.classList.remove('selected');
            cardEl.classList.remove('flipped');
            selectedCards = selectedCards.filter(c => c.cardData.id !== cardData.id);
        } else {
            // Check limit
            if (selectedCards.length >= currentMode) {
                alert(`Máximo ${currentMode} cartas permitidas en este modo.`);
                return;
            }
            // Select
            cardEl.classList.add('selected');
            cardEl.classList.add('flipped'); // Reveal on selection
            selectedCards.push({ cardData, isReversed });
        }
        updateStatus();
    }

    // --- Prompt Generation ---

    function generatePrompt() {
        if (selectedCards.length === 0) {
            alert("Por favor selecciona al menos una carta.");
            return;
        }

        const context = document.getElementById('user-context').value;
        const output = document.getElementById('prompt-output');

        let prompt = `CONSULTA DE TAROT\n\n`;
        prompt += `Cartas seleccionadas:\n`;

        selectedCards.forEach((item, index) => {
            const pos = item.isReversed ? "Reversa" : "Normal";
            prompt += `${index + 1}. ${item.cardData.name_es} - Posición: ${pos}\n`;
        });

        prompt += `\nTipo de lectura: ${currentMode} cartas\n`;
        if (context.trim()) {
            prompt += `Contexto de la consulta: ${context}\n`;
        } else {
            prompt += `Contexto de la consulta: Sin contexto específico.\n`;
        }

        prompt += `\nPor favor, interpreta esta lectura considerando:
- El significado tradicional de cada carta en su posición
- Las relaciones entre las cartas en este arreglo específico
- Cualquier patrón o tema que observes como sinergia, números, sombras
- Proporciona una interpretación integral de la lectura
- Se neutral pero trata de integrar el consejo con acciones útiles`;

        output.value = prompt;
    }

    function copyPrompt() {
        const output = document.getElementById('prompt-output');
        output.select();
        document.execCommand('copy');
        alert("Prompt copiado al portapapeles");
    }

    // --- Modal ---
    const modal = document.getElementById('card-modal');
    const closeModal = document.querySelector('.close-modal');

    closeModal.onclick = function () {
        modal.style.display = "none";
    }

    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }

    function showModal(data) {
        document.getElementById('modal-title').textContent = data.name_es + " / " + data.name_en;
        document.getElementById('modal-type').textContent = data.type === 'major' ? "Arcano Mayor" : `Arcano Menor - ${capitalize(data.suit)}`;
        document.getElementById('modal-keywords-up').textContent = data.keywords_up;
        document.getElementById('modal-keywords-rev').textContent = data.keywords_rev;
        // document.getElementById('modal-img').src = data.img; 

        modal.style.display = "block";
    }

    function capitalize(s) {
        if (!s) return '';
        return s.charAt(0).toUpperCase() + s.slice(1);
    }
});
