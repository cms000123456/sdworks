// ===============================================
// SDWorks - APPLICATION LOGIC
// ===============================================

class StableDiffusionApp {
    constructor() {
        this.apiEndpoint = ''; // Will be set in loadSettings

        // Style Definitions
        this.PROMPT_STYLES = {
            "None": { prompt: "", negative: "" },
            "Cinematic": {
                prompt: ", cinematic lighting, highly detailed, photorealistic, 8k, depth of field, anamorphic lens, film grain",
                negative: ", cartoon, anime, sketches, low quality, worst quality"
            },
            "Anime": {
                prompt: ", anime style, vibrant colors, studio ghibli, makoto shinkai, highly detailed, sharp focus",
                negative: ", photorealistic, realistic, photograph, 3d render"
            },
            "Digital Art": {
                prompt: ", concept art, digital painting, sharp focus, artstation, 8k, illustration, smooth",
                negative: ", photo, grain, noise"
            },
            "Photographic": {
                prompt: ", 35mm photograph, film, bokeh, professional, 4k, highly detailed, raw photo",
                negative: ", drawing, painting, illustration, render, 3d"
            },
            "Fantasy": {
                prompt: ", fantasy art, ethereal, vibrant, magical, intricate details, oil painting style",
                negative: ", sci-fi, modern, photograph"
            },
            "Cyberpunk": {
                prompt: ", cyberpunk, neon lights, futuristic, rain, high tech, highly detailed, urban, sci-fi",
                negative: ", historical, rustic, nature"
            },
            "3D Model": {
                prompt: ", 3d render, unreal engine 5, octane render, physically based rendering, highly detailed",
                negative: ", 2d, sketch, painting, cartoon"
            },
            "Pixel Art": {
                prompt: ", pixel art, low res, 8 bit, 16 bit, retro games, sharp edges",
                negative: ", blur, high resolution, realistic, vector"
            },
            "Line Art": {
                prompt: ", line art, monochrome, ink sketch, minimalist, clean lines",
                negative: ", color, shading, realistic, photo"
            },
            "Comic Book": {
                prompt: ", comic book style, bold outlines, vibrant colors, dynamic action",
                negative: ", realistic, photo, soft"
            },
            "Oil Painting": {
                prompt: ", oil painting, heavy strokes, textured, canvas, classic art",
                negative: ", digital art, photo, flat"
            },
            "Watercolor": {
                prompt: ", watercolor painting, soft edges, artistic, bleeding colors, wet on wet",
                negative: ", sharp, photo, digital art"
            },
            "Noir": {
                prompt: ", film noir, black and white, high contrast, dramatic shadows, moody",
                negative: ", color, bright, happy"
            },
            "Vintage": {
                prompt: ", vintage photo, antique, sepia, retro, 1950s, grain, damage",
                negative: ", modern, digital, clean"
            },
            "Neon": {
                prompt: ", neon style, glowing, synthwave, dark background, vibrant lights",
                negative: ", day, natural light, dull"
            },
            "Minimalist": {
                prompt: ", minimalist, simple, clean lines, negative space, icon style",
                negative: ", detailed, cluttered, realistic"
            }
        };

        // Prompt Helper Data
        this.HELPER_DATA = {
            "Subject": ["Cybernetic Samurai", "Astronaut in Jungle", "Vampire Queen", "Ancient Dragon", "Neon Cat", "Steampunk Inventor", "Medieval Knight", "Cyberpunk Girl", "Forest Spirit", "Mechanical Owl"],
            "Artstyle": ["Oil Painting", "Concept Art", "Water Color", "Synthwave", "Impressionism", "Ukiyo-e", "Pop Art", "Charcoal Sketch", "Double Exposure", "Voxel Art"],
            "Lighting": ["Volumetric Lighting", "Golden Hour", "Soft Glow", "Cinematic Shadows", "Neon Rim Light", "HDR", "Global Illumination", "Moonlight", "Dramatic Backlight", "Soft Bokeh"],
            "Camera": ["Macro Shot", "Wide Angle", "Fish Eye", "Aerial View", "Close Up", "Low Angle", "Side View", "Tilt Shift", "80mm Lens", "Action Shot"],
            "Mood": ["Ethereal", "Melancholy", "Epic", "Cozy", "Surreal", "Grim", "Energetic", "Peaceful", "Mysterious", "Nostalgic"],
            "Quality": ["Masterpiece", "Highly Detailed", "8k Resolution", "Intricate Details", "Professional Quality", "Sharp Focus", "Hyperrealistic", "Award Winning", "Render Artifacts", "Soft Skin"]
        };

        // State variables
        this.selectedLoras = [];
        this.isGenerating = false;
        this.initImageBase64 = null;
        this.isImg2Img = false;
        this.savedPrompts = JSON.parse(localStorage.getItem('sd_saved_prompts') || '[]');
        this.generatedImages = [];
        this.currentModelArch = 'sd15'; // Default
        this.availableModelsList = [];
        this.lastParams = null; // Store last generation params for filenames
        this.availableLoras = [];
        this.selectedLoras = []; // { name, weight }
        this.currentlyLoadedModel = null; // Track actual loaded model from API

        this.initializeElements();
        this.setupErrorTracking();
        this.attachEventListeners();
        this.loadSettings();
        this.loadDefaults(); // Load user defaults
        this.updateCharCount();

        // Initial data fetch
        this.loadModels();
        this.fetchLoras();
        this.loadLorasFromStorage();
        if (this.savedPrompts.length > 0) {
            console.log(`Loaded ${this.savedPrompts.length} saved prompts`);
        }

        this.initStyleDropdown();
    }

    // Capture errors to show in UI
    setupErrorTracking() {
        window.addEventListener('error', (e) => {
            this.showNotification(`JS Error: ${e.message}`, 'error');
        });
        window.addEventListener('unhandledrejection', (e) => {
            this.showNotification(`Promise Error: ${e.reason}`, 'error');
        });
    }

    // Initialize DOM elements
    initializeElements() {
        // Form elements
        this.promptInput = document.getElementById('prompt');
        this.negativePromptInput = document.getElementById('negativePrompt');
        this.stepsInput = document.getElementById('stepsInput');
        this.stepsValue = document.getElementById('stepsValue');
        this.cfgInput = document.getElementById('cfgScale');
        this.cfgValue = document.getElementById('cfgScaleValue');
        this.samplerSelect = document.getElementById('samplerSelect');
        this.widthInput = document.getElementById('width');
        this.heightInput = document.getElementById('height');
        this.seedInput = document.getElementById('seed');
        this.batchSizeSelect = document.getElementById('batchSize');
        this.apiEndpointInput = document.getElementById('apiEndpoint');
        this.refinementPromptInput = document.getElementById('refinementPrompt');
        this.clearRefinementBtn = document.getElementById('clearRefinementBtn');

        // Model elements
        this.modelSelect = document.getElementById('modelSelect');
        this.modelSelect = document.getElementById('modelSelect');
        this.modelStatus = document.getElementById('modelStatus');
        this.modelSearchInput = document.getElementById('modelSearchInput');

        // Buttons
        this.generateBtn = document.getElementById('generateBtn');
        this.randomSeedBtn = document.getElementById('randomSeedBtn');
        this.saveDefaultsBtn = document.getElementById('saveDefaultsBtn');
        this.saveStyleBtn = document.getElementById('saveStyleBtn');

        // Display elements
        this.emptyState = document.getElementById('emptyState');
        this.galleryGrid = document.getElementById('galleryGrid');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.generationInfo = document.getElementById('generationInfo');
        this.generationDetails = document.getElementById('generationDetails');
        this.charCount = document.getElementById('charCount');

        // Modal
        this.imageModal = document.getElementById('imageModal');
        this.modalImage = document.getElementById('modalImage');
        this.closeModalBtn = document.getElementById('closeModal');

        // Model Upload
        this.modelUploadInput = document.getElementById('modelUploadInput');
        this.browseModelBtn = document.getElementById('browseModelBtn');
        this.uploadModelBtn = document.getElementById('uploadModelBtn');
        this.uploadProgressContainer = document.getElementById('uploadProgressContainer');
        this.uploadProgressBar = document.getElementById('uploadProgressBar');
        this.uploadProgressText = document.getElementById('uploadProgressText');

        // Civitai elements
        this.civitaiSearchInput = document.getElementById('civitaiSearchInput');
        this.civitaiSearchBtn = document.getElementById('civitaiSearchBtn');
        this.civitaiResults = document.getElementById('civitaiResults');
        this.tabBtns = document.querySelectorAll('.tab-btn');
        this.tabContents = document.querySelectorAll('.tab-content');

        // Direct Download elements
        this.directDownloadUrl = document.getElementById('directDownloadUrl');
        this.directDownloadType = document.getElementById('directDownloadType');
        this.directDownloadBtn = document.getElementById('directDownloadBtn');

        // LoRA Elements
        this.addLoraBtn = document.getElementById('addLoraBtn');
        this.activeLorasList = document.getElementById('activeLorasList');
        this.noLorasMessage = document.getElementById('noLorasMessage');
        this.loraModal = document.getElementById('loraModal');
        this.closeLoraModalBtn = document.getElementById('closeLoraModalBtn');
        this.loraSearchInput = document.getElementById('loraSearchInput');
        this.loraListDisplay = document.getElementById('loraList');

        // Img2Img Elements
        this.modeTxt2Img = document.getElementById('modeTxt2Img');
        this.modeImg2Img = document.getElementById('modeImg2Img');
        this.img2imgContainer = document.getElementById('img2imgContainer');
        this.dropZone = document.getElementById('dropZone');
        this.initImageInput = document.getElementById('initImageInput');
        this.initImagePreview = document.getElementById('initImagePreview');
        this.removeInitImageBtn = document.getElementById('removeInitImageBtn');
        this.denoisingStrength = document.getElementById('denoisingStrength');
        this.denoisingStrengthValue = document.getElementById('denoisingStrengthValue');

        // Saved Prompts Elements
        this.savePromptBtn = document.getElementById('savePromptBtn');
        this.loadPromptBtn = document.getElementById('loadPromptBtn');
        this.promptsModal = document.getElementById('promptsModal');
        this.closePromptsModalBtn = document.getElementById('closePromptsModalBtn');
        this.promptsList = document.getElementById('promptsList');

        // Styles
        this.styleSelect = document.getElementById('styleSelect');

        // Prompt Helper
        this.toggleHelperBtn = document.getElementById('toggleHelperBtn');
        this.promptHelper = document.getElementById('promptHelper');
        this.helperTags = document.getElementById('helperTags');
        this.helperTabs = document.querySelectorAll('.helper-tab');

        // Prompt Refinement
        this.interrogateBtn = document.getElementById('interrogateBtn');
        this.magicPromptBtn = document.getElementById('magicPromptBtn');

        // Presets
        this.presetPortraitBtn = document.getElementById('presetPortrait');
        this.presetLandscapeBtn = document.getElementById('presetLandscape');
    }

    // Attach event listeners
    attachEventListeners() {
        // Prompt character count
        this.promptInput.addEventListener('input', () => this.updateCharCount());
        this.refinementPromptInput.addEventListener('input', () => this.updateCharCount());

        if (this.clearRefinementBtn) {
            this.clearRefinementBtn.addEventListener('click', () => {
                this.refinementPromptInput.value = '';
                this.updateCharCount();
                this.showNotification('Refinements cleared', 'info');
            });
        }

        // Slider value updates
        this.stepsInput.addEventListener('input', () => {
            this.stepsValue.textContent = this.stepsInput.value;
        });

        this.cfgInput.addEventListener('input', () => {
            this.cfgValue.textContent = parseFloat(this.cfgInput.value).toFixed(1);
        });

        // Random seed button
        this.randomSeedBtn.addEventListener('click', () => {
            this.seedInput.value = Math.floor(Math.random() * 2147483647);
        });

        // Generate button
        this.generateBtn.addEventListener('click', () => this.generateImages());

        // Save Defaults
        if (this.saveDefaultsBtn) {
            this.saveDefaultsBtn.addEventListener('click', () => this.saveDefaults());
        }

        // Save Style
        if (this.saveStyleBtn) {
            this.saveStyleBtn.addEventListener('click', () => this.saveCurrentAsPreset());
        }

        // API endpoint save
        this.apiEndpointInput.addEventListener('change', () => {
            this.saveSettings();
            this.loadModels(); // Reload models when endpoint changes
        });

        // Model selection
        this.modelSelect.addEventListener('change', () => {
            if (this.modelSelect.value === '_refresh') {
                this.filterModels(''); // Reset filter
                if (this.modelSearchInput) this.modelSearchInput.value = '';
                this.loadModels();
            } else {
                this.switchModel();
            }
        });

        if (this.modelSearchInput) {
            this.modelSearchInput.addEventListener('input', (e) => this.filterModels(e.target.value));
        }

        // LoRA Events
        if (this.addLoraBtn) this.addLoraBtn.addEventListener('click', () => this.openLoraModal());
        if (this.closeLoraModalBtn) this.closeLoraModalBtn.addEventListener('click', () => this.closeLoraModal());
        if (this.loraSearchInput) this.loraSearchInput.addEventListener('input', () => this.searchLoras());

        // Close lora modal on outside click
        if (this.loraModal) {
            this.loraModal.addEventListener('click', (e) => {
                if (e.target === this.loraModal) this.closeLoraModal();
            });
        }

        // Modal controls
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        this.imageModal.addEventListener('click', (e) => {
            if (e.target === this.imageModal) this.closeModal();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to generate
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                this.generateImages();
            }
            // Escape to close modal
            if (e.key === 'Escape') {
                this.closeModal();
            }
        });

        // Model Upload listeners
        this.browseModelBtn.addEventListener('click', () => this.modelUploadInput.click());
        this.uploadModelBtn.addEventListener('click', () => this.handleModelUpload());

        // Civitai listeners
        this.civitaiSearchBtn.addEventListener('click', () => this.searchCivitai());
        this.civitaiSearchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchCivitai();
        });

        // Direct Download listener
        if (this.directDownloadBtn) {
            this.directDownloadBtn.addEventListener('click', () => this.handleDirectDownload());
        }

        // Tab switching
        this.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });

        // Presets
        this.presetPortraitBtn.addEventListener('click', () => this.applyPreset('portrait'));
        this.presetLandscapeBtn.addEventListener('click', () => this.applyPreset('landscape'));

        // Img2Img Mode Switching
        this.modeTxt2Img.addEventListener('change', () => this.setMode('txt2img'));
        this.modeImg2Img.addEventListener('change', () => this.setMode('img2img'));

        // Image Upload
        this.dropZone.addEventListener('click', () => this.initImageInput.click());
        this.initImageInput.addEventListener('change', (e) => this.handleImageSelect(e.target.files[0]));

        this.dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.dropZone.classList.add('dragover');
        });
        this.dropZone.addEventListener('dragleave', () => this.dropZone.classList.remove('dragover'));
        this.dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            this.dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                this.handleImageSelect(e.dataTransfer.files[0]);
            }
        });

        this.removeInitImageBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.removeInitImage();
        });

        this.denoisingStrength.addEventListener('input', () => {
            this.denoisingStrengthValue.textContent = this.denoisingStrength.value;
        });

        // Saved Prompts
        this.savePromptBtn.addEventListener('click', () => this.saveCurrentPrompt());
        this.loadPromptBtn.addEventListener('click', () => this.openPromptsModal());
        if (this.closePromptsModalBtn) this.closePromptsModalBtn.addEventListener('click', () => this.closePromptsModal());
        if (this.promptsModal) {
            this.promptsModal.addEventListener('click', (e) => {
                if (e.target === this.promptsModal) this.closePromptsModal();
            });
        }

        // Prompt Helper Events
        if (this.toggleHelperBtn) {
            this.toggleHelperBtn.addEventListener('click', () => this.togglePromptHelper());
        }

        if (this.interrogateBtn) {
            this.interrogateBtn.addEventListener('click', () => this.interrogateImage());
        }

        if (this.magicPromptBtn) {
            this.magicPromptBtn.addEventListener('click', () => this.magicExpandPrompt());
        }

        this.helperTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                this.helperTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                this.renderHelperTags(tab.dataset.cat);
            });
        });

        // Initialize helper with first category
        this.renderHelperTags('Subject');
    }

    togglePromptHelper() {
        this.promptHelper.classList.toggle('hidden');
        if (!this.promptHelper.classList.contains('hidden')) {
            this.toggleHelperBtn.classList.add('btn-primary');
            this.toggleHelperBtn.classList.remove('btn-secondary');
        } else {
            this.toggleHelperBtn.classList.add('btn-secondary');
            this.toggleHelperBtn.classList.remove('btn-primary');
        }
    }

    renderHelperTags(category) {
        if (!this.helperTags || !this.HELPER_DATA[category]) return;

        this.helperTags.innerHTML = '';
        this.HELPER_DATA[category].forEach(tag => {
            const tagEl = document.createElement('span');
            tagEl.className = 'helper-tag';
            tagEl.textContent = tag;
            tagEl.addEventListener('click', () => this.addTagToPrompt(tag));
            this.helperTags.appendChild(tagEl);
        });
    }

    addTagToPrompt(tag) {
        let currentPrompt = this.refinementPromptInput.value.trim();

        if (currentPrompt === '') {
            currentPrompt = tag;
        } else {
            // Check if tag already exists (fuzzy check)
            if (currentPrompt.toLowerCase().includes(tag.toLowerCase())) {
                this.showNotification(`Tag "${tag}" already in prompt`, 'info');
                return;
            }

            // Add comma if needed
            if (currentPrompt.endsWith(',')) {
                currentPrompt += ' ' + tag;
            } else {
                currentPrompt += ', ' + tag;
            }
        }

        this.refinementPromptInput.value = currentPrompt;
        this.updateCharCount();
        this.showNotification(`Added "${tag}" to refinements`, 'success');

        // Brief visual feedback
        setTimeout(() => this.refinementPromptInput.focus(), 50);
    }

    async interrogateImage() {
        if (!this.initImageBase64) {
            this.showNotification('Please upload an image first', 'warning');
            return;
        }

        this.showNotification('Interrogating image...', 'info');
        const originalBtnText = this.interrogateBtn.textContent;
        this.interrogateBtn.disabled = true;
        this.interrogateBtn.textContent = '🔍 ...';

        try {
            const response = await fetch(`${this.apiEndpoint}/sdapi/v1/interrogate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: this.initImageBase64 })
            });

            if (!response.ok) throw new Error('Interrogation failed');

            const data = await response.json();
            if (data.prompt) {
                // Interrogation fills the refinements by default, or the subject if empty
                if (this.promptInput.value.trim() === '') {
                    this.promptInput.value = data.prompt;
                } else {
                    this.refinementPromptInput.value = data.prompt;
                }
                this.updateCharCount();
                this.showNotification('Image described!', 'success');
            }
        } catch (error) {
            console.error(error);
            this.showNotification('Interrogation failed', 'error');
        } finally {
            this.interrogateBtn.disabled = false;
            this.interrogateBtn.textContent = originalBtnText;
        }
    }

    async magicExpandPrompt() {
        const prompt = this.promptInput.value.trim();
        if (!prompt) {
            this.showNotification('Enter a short prompt first', 'warning');
            return;
        }

        this.showNotification('Expanding prompt...', 'info');
        const originalBtnText = this.magicPromptBtn.textContent;
        this.magicPromptBtn.disabled = true;
        this.magicPromptBtn.textContent = '🪄 ...';

        try {
            const response = await fetch(`${this.apiEndpoint}/sdapi/v1/magic-prompt`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt })
            });

            if (!response.ok) throw new Error('Expansion failed');

            const data = await response.json();
            if (data.prompt) {
                // Magic expand fills the refinements part
                this.refinementPromptInput.value = data.prompt;
                this.updateCharCount();
                this.showNotification('Refinements magically updated!', 'success');
            }
        } catch (error) {
            console.error(error);
            this.showNotification('Expansion failed', 'error');
        } finally {
            this.magicPromptBtn.disabled = false;
            this.magicPromptBtn.textContent = originalBtnText;
        }
    }

    setMode(mode) {
        this.isImg2Img = mode === 'img2img';
        if (this.isImg2Img) {
            this.img2imgContainer.style.display = 'block';
        } else {
            this.img2imgContainer.style.display = 'none';
        }
    }

    handleImageSelect(file) {
        if (!file || !file.type.startsWith('image/')) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            this.initImageBase64 = e.target.result.split(',')[1]; // Remove data:image/png;base64, prefix
            this.initImagePreview.src = e.target.result;
            this.initImagePreview.style.display = 'block';
            this.initImagePreview.style.display = 'block';
            this.dropZone.querySelector('.drop-zone-content').style.display = 'none';
            this.removeInitImageBtn.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    removeInitImage() {
        this.initImageBase64 = null;
        this.initImagePreview.src = '';
        this.initImagePreview.style.display = 'none';
        this.initImagePreview.src = '';
        this.initImagePreview.style.display = 'none';
        this.dropZone.querySelector('.drop-zone-content').style.display = 'block';
        this.removeInitImageBtn.style.display = 'none';
        this.initImageInput.value = '';
    }

    // Update character count
    updateCharCount() {
        const baseCount = this.promptInput.value.length;
        const refinedCount = this.refinementPromptInput.value.length;
        this.charCount.textContent = baseCount + refinedCount;
    }

    // Load settings from localStorage
    loadSettings() {
        const savedEndpoint = localStorage.getItem('sd_api_endpoint');

        // If we have a saved endpoint, use it
        if (savedEndpoint !== null && savedEndpoint !== '') {
            this.apiEndpointInput.value = savedEndpoint;
        } else {
            // Default to empty (relative) which uses the Nginx proxy
            this.apiEndpointInput.value = '';
        }

        this.apiEndpoint = this.apiEndpointInput.value.replace(/\/$/, ''); // Remove trailing slash

        // Log for debugging (user can check console)
        console.log(`Stable Diffusion App initialized. API Endpoint: "${this.apiEndpoint || '(relative proxy)'}"`);

        // Load models after settings are loaded
        this.loadModels();
        this.loadDefaults(); // Load user defaults
    }

    // Save settings to localStorage
    saveSettings() {
        const settings = {
            apiEndpoint: this.apiEndpointInput.value
        };
        localStorage.setItem('sd_settings', JSON.stringify(settings));
        this.showNotification('Settings saved', 'success');
    }

    // Save current parameters as default
    saveDefaults() {
        const defaults = {
            steps: this.stepsInput.value,
            cfg: this.cfgInput.value,
            width: this.widthInput.value,
            height: this.heightInput.value,
            sampler: this.samplerSelect.value,
            style: this.styleSelect.value,
            batchSize: this.batchSizeSelect.value,
            negativePrompt: this.negativePromptInput.value
        };
        localStorage.setItem('sd_user_defaults', JSON.stringify(defaults));
        this.showNotification('Parameters & Negative Prompt saved!', 'success');
    }

    // Load user defaults
    loadDefaults() {
        try {
            const defaults = JSON.parse(localStorage.getItem('sd_user_defaults'));
            if (!defaults) return;

            if (defaults.steps) {
                this.stepsInput.value = defaults.steps;
                this.stepsValue.textContent = defaults.steps;
            }
            if (defaults.cfg) {
                this.cfgInput.value = defaults.cfg;
                this.cfgValue.textContent = parseFloat(defaults.cfg).toFixed(1);
            }
            if (defaults.width) this.widthInput.value = defaults.width;
            if (defaults.height) this.heightInput.value = defaults.height;
            if (defaults.sampler) this.samplerSelect.value = defaults.sampler;
            if (defaults.style) this.styleSelect.value = defaults.style;
            if (defaults.batchSize) this.batchSizeSelect.value = defaults.batchSize;
            if (defaults.negativePrompt !== undefined) this.negativePromptInput.value = defaults.negativePrompt;

            console.log('Loaded user defaults:', defaults);
        } catch (e) {
            console.error('Error loading defaults:', e);
        }
    }

    // Load custom styles from localStorage
    loadUserStyles() {
        try {
            const userStyles = JSON.parse(localStorage.getItem('sd_user_styles') || '{}');
            Object.assign(this.PROMPT_STYLES, userStyles);
        } catch (e) {
            console.error('Error loading user styles:', e);
        }
    }

    // Save current configuration as a new style preset
    saveCurrentAsPreset() {
        const name = prompt("Enter a name for this style preset:");
        if (!name) return;

        if (this.PROMPT_STYLES[name]) {
            if (!confirm(`Style '${name}' already exists. Overwrite?`)) return;
        }

        const newStyle = {
            prompt: this.promptInput.value ? ", " + this.promptInput.value : "",
            negative: this.negativePromptInput.value ? ", " + this.negativePromptInput.value : ""
        };

        // Save to PROMPT_STYLES
        this.PROMPT_STYLES[name] = newStyle;

        // Persist to localStorage
        try {
            const userStyles = JSON.parse(localStorage.getItem('sd_user_styles') || '{}');
            userStyles[name] = newStyle;
            localStorage.setItem('sd_user_styles', JSON.stringify(userStyles));

            this.showNotification(`Style '${name}' saved!`, 'success');

            // Reload dropdown
            this.initStyleDropdown();
            this.styleSelect.value = name; // Select it
        } catch (e) {
            console.error('Error saving style:', e);
            this.showNotification('Failed to save style', 'error');
        }
    }

    initStyleDropdown() {
        // Load user styles first
        this.loadUserStyles();

        if (!this.styleSelect) return;
        this.styleSelect.innerHTML = '';
        Object.keys(this.PROMPT_STYLES).forEach(style => {
            const option = document.createElement('option');
            option.value = style;
            option.textContent = style;
            this.styleSelect.appendChild(option);
        });
    }

    getParameters() {
        if (this.isImg2Img && !this.initImageBase64) {
            this.showNotification('Please upload an image for Img2Img mode', 'warning');
            return null;
        }

        let basePrompt = this.promptInput.value.trim();
        let refinedPrompt = this.refinementPromptInput.value.trim();

        // Merge prompts
        let finalPrompt = basePrompt;
        if (refinedPrompt) {
            if (finalPrompt && !finalPrompt.endsWith(',')) finalPrompt += ', ';
            finalPrompt += refinedPrompt;
        }

        let finalNegative = this.negativePromptInput.value.trim();

        // Apply Style
        if (this.styleSelect && this.styleSelect.value !== "None") {
            const style = this.PROMPT_STYLES[this.styleSelect.value];
            if (style) {
                if (finalPrompt && !finalPrompt.endsWith(',')) finalPrompt += ', ';
                finalPrompt += style.prompt.startsWith(',') ? style.prompt.substring(1) : style.prompt;
                finalNegative += style.negative;
            }
        }

        return {
            prompt: finalPrompt,
            negative_prompt: finalNegative,
            steps: parseInt(this.stepsInput.value),
            cfg_scale: parseFloat(this.cfgInput.value),
            width: parseInt(this.widthInput.value),
            height: parseInt(this.heightInput.value),
            seed: parseInt(this.seedInput.value),
            batch_size: parseInt(this.batchSizeSelect.value),
            n_iter: 1,
            sampler_name: this.samplerSelect ? this.samplerSelect.value : "DPM++ 2M Karras",
            loras: this.selectedLoras.length > 0 ? this.selectedLoras : null,
            init_images: this.isImg2Img ? [this.initImageBase64] : null,
            denoising_strength: this.isImg2Img ? parseFloat(this.denoisingStrength ? this.denoisingStrength.value : 0.75) : 0.75
        };
    }

    // Validate parameters
    validateParameters(params) {
        if (!params.prompt) {
            this.showNotification('Please enter a prompt', 'error');
            return false;
        }

        if (params.prompt === undefined || params.prompt === '') {
            this.showNotification('Please enter a prompt', 'error');
            return false;
        }

        // Allow empty endpoint (relative path)
        return true;

        return true;
    }

    // Generate images
    async generateImages() {
        if (this.isGenerating) return;

        const params = this.getParameters();
        if (!this.validateParameters(params)) return;

        this.setGenerating(true);
        const startTime = Date.now();

        try {
            // Call Stable Diffusion API
            const response = await this.callAPI(params);

            if (response.images && response.images.length > 0) {
                const endTime = Date.now();
                const duration = (endTime - startTime) / 1000;

                this.lastParams = params; // Save params for smart filenames
                // Use parameters from API response if available (contains actual seed)
                this.displayImages(response.images, response.parameters || params, duration);
                this.showNotification(`Generated ${response.images.length} image(s) in ${duration.toFixed(2)}s`, 'success');
            } else {
                throw new Error('No images returned from API');
            }
        } catch (error) {
            console.error('Generation error:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.setGenerating(false);
        }
    }

    // Call Stable Diffusion API
    async callAPI(params) {
        // Try AUTOMATIC1111 format first
        const endpoint = `${this.apiEndpoint}/sdapi/v1/txt2img`;

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error(`API request failed: ${response.status} ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            // If CORS or network error, provide helpful message
            if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
                throw new Error('Cannot connect to API. Make sure CORS is enabled on your Stable Diffusion backend and the URL is correct.');
            }
            throw error;
        }
    }

    // Display generated images
    displayImages(images, params, duration = 0) {
        // Hide empty state, show gallery
        this.emptyState.classList.add('hidden');
        this.galleryGrid.classList.remove('hidden');

        // Clear previous images
        this.galleryGrid.innerHTML = '';
        this.generatedImages = [];

        // Add new images
        images.forEach((base64Image, index) => {
            const imageData = `data:image/png;base64,${base64Image}`;
            this.generatedImages.push(imageData);

            const galleryItem = this.createGalleryItem(imageData, index);
            this.galleryGrid.appendChild(galleryItem);
        });

        // Show generation info
        this.showGenerationInfo(params, images.length, duration);
    }

    // Create gallery item element
    createGalleryItem(imageData, index) {
        const item = document.createElement('div');
        item.className = 'gallery-item';

        const img = document.createElement('img');
        img.src = imageData;
        img.alt = `Generated image ${index + 1}`;
        img.loading = 'lazy';

        // Click to view full size
        img.addEventListener('click', () => this.openModal(imageData));

        const overlay = document.createElement('div');
        overlay.className = 'gallery-item-overlay';

        const actions = document.createElement('div');
        actions.className = 'gallery-item-actions';

        // Download button
        const downloadBtn = this.createIconButton('💾', 'Download', () => {
            const filename = this.getSmartFilename(this.lastParams, index);
            this.downloadImage(imageData, filename);
        });

        // View full size button
        const viewBtn = this.createIconButton('🔍', 'View Full Size', () => {
            this.openModal(imageData);
        });

        actions.appendChild(viewBtn);
        actions.appendChild(downloadBtn);
        overlay.appendChild(actions);

        item.appendChild(img);
        item.appendChild(overlay);

        return item;
    }

    // Create icon button
    createIconButton(icon, title, onClick) {
        const btn = document.createElement('button');
        btn.className = 'icon-btn';
        btn.innerHTML = icon;
        btn.title = title;
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            onClick();
        });
        return btn;
    }

    // Download image
    // Helper to convert dataURL to Blob
    dataURLtoBlob(dataurl) {
        try {
            const arr = dataurl.split(',');
            const mime = arr[0].match(/:(.*?);/)[1];
            const bstr = atob(arr[1]);
            let n = bstr.length;
            const u8arr = new Uint8Array(n);
            while (n--) {
                u8arr[n] = bstr.charCodeAt(n);
            }
            return new Blob([u8arr], { type: mime });
        } catch (e) {
            console.error("Failed to convert dataURL to blob", e);
            return null;
        }
    }

    // Download image
    downloadImage(dataUrl, filename) {
        let downloadUrl = dataUrl;
        let isBlob = false;

        // Try converting to blob for better browser support
        const blob = this.dataURLtoBlob(dataUrl);
        if (blob) {
            downloadUrl = URL.createObjectURL(blob);
            isBlob = true;
        }

        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();

        // Delay removal to ensure browser processes the click
        setTimeout(() => {
            document.body.removeChild(link);
            if (isBlob) {
                URL.revokeObjectURL(downloadUrl);
            }
        }, 2000);

        this.showNotification('Image downloaded', 'success');
    }

    // Helper to generate smart filename
    getSmartFilename(params, index = 0) {
        if (!params) return `sd-image-${Date.now()}-${index}.png`;

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        const modelName = this.modelSelect.value.split('/').pop().split('\\').pop().replace('.safetensors', '').replace('.ckpt', '');

        // Clean prompt for filename (first 30 chars, alphanumeric only)
        const cleanPrompt = params.prompt
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '_')
            .replace(/__+/g, '_')
            .slice(0, 30);

        return `${timestamp}_${modelName}_${cleanPrompt}_${index}.png`;
    }

    // Download all images in the current batch
    downloadAllImages() {
        if (!this.generatedImages || this.generatedImages.length === 0) return;

        this.showNotification(`Starting batch download of ${this.generatedImages.length} images...`, 'info');

        this.generatedImages.forEach((imageData, index) => {
            // Add a small delay between downloads to prevent browser blocking
            setTimeout(() => {
                const filename = this.getSmartFilename(this.lastParams, index);
                this.downloadImage(imageData, filename);
            }, index * 200);
        });
    }

    // Open image modal
    openModal(imageData) {
        this.modalImage.src = imageData;
        this.imageModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    // Close image modal
    closeModal() {
        this.imageModal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    // Show generation info
    showGenerationInfo(params, imageCount, duration = 0) {
        this.generationInfo.classList.remove('hidden');

        const seed = params.seed;

        this.generationDetails.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
                <div><strong>Images:</strong> ${imageCount}</div>
                <div><strong>Steps:</strong> ${params.steps}</div>
                <div><strong>CFG Scale:</strong> ${params.cfg_scale}</div>
                <div><strong>Sampler:</strong> ${params.sampler_name || 'DPM++ 2M Karras'}</div>
                <div><strong>Size:</strong> ${params.width}×${params.height}</div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <strong>Seed:</strong> ${seed}
                    <button type="button" class="btn btn-secondary btn-xs" id="reuseSeedBtn" title="Reuse this seed" style="padding: 0 4px; font-size: 0.7rem;">♻️</button>
                </div>
                <div><strong>Batch:</strong> ${params.batch_size}</div>
                <div><strong>Time:</strong> ${duration ? duration.toFixed(2) + 's' : 'N/A'}</div>
                <div><strong>Speed:</strong> ${duration && params.steps ? (params.steps / duration).toFixed(2) + ' it/s' : 'N/A'}</div>
            </div>
            ${params.prompt ? `<div style="margin-top: 0.75rem;"><strong>Prompt:</strong><br>${this.escapeHtml(params.prompt)}</div>` : ''}
            ${params.negative_prompt ? `<div style="margin-top: 0.5rem; opacity: 0.7; font-size: 0.85rem;"><strong>Negative Prompt:</strong><br>${this.escapeHtml(params.negative_prompt)}</div>` : ''}
            
            <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); display: flex; justify-content: flex-end;">
                <button type="button" class="btn btn-secondary btn-sm" id="downloadAllBtn" style="padding: 0.5rem 1rem;">
                    📦 Download All ${imageCount} Images
                </button>
            </div>
        `;

        // Add event listener for download all button
        const downloadAllBtn = document.getElementById('downloadAllBtn');
        if (downloadAllBtn) {
            downloadAllBtn.addEventListener('click', () => this.downloadAllImages());
        }

        // Add event listener for reuse seed button
        const reuseSeedBtn = document.getElementById('reuseSeedBtn');
        if (reuseSeedBtn) {
            reuseSeedBtn.addEventListener('click', () => {
                this.seedInput.value = seed;
                this.showNotification('Seed applied to settings', 'success');
            });
        }
    }

    // Set generating state
    setGenerating(generating) {
        this.isGenerating = generating;

        if (generating) {
            this.loadingOverlay.classList.remove('hidden');
            this.generateBtn.disabled = true;
            document.getElementById('generateBtnText').textContent = 'Generating...';
            document.getElementById('generateBtnSpinner').classList.remove('hidden');
        } else {
            this.loadingOverlay.classList.add('hidden');
            this.generateBtn.disabled = false;
            document.getElementById('generateBtnText').textContent = '✨ Generate Images';
            document.getElementById('generateBtnSpinner').classList.add('hidden');
        }
    }

    // Show notification
    showNotification(message, type = 'success') {
        const container = document.getElementById('notificationContainer');

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <strong>${type === 'success' ? '✓' : type === 'error' ? '✕' : '⚠'}</strong>
            ${this.escapeHtml(message)}
        `;

        container.appendChild(notification);

        // Auto remove after 4 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 300ms ease-in-out';
            setTimeout(() => {
                container.removeChild(notification);
            }, 300);
        }, 4000);
    }

    // Escape HTML
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    filterModels(query) {
        if (!this.availableModelsList) return;

        if (!query) {
            this.renderModelOptions(this.availableModelsList);
            // Try to restore selection
            if (this.currentlyLoadedModel) {
                this.modelSelect.value = this.currentlyLoadedModel;
                this.updateModelHighlight();
            }
            return;
        }

        const lowerQuery = query.toLowerCase();
        const filtered = this.availableModelsList.filter(m =>
            (m.title && m.title.toLowerCase().includes(lowerQuery)) ||
            (m.model_name && m.model_name.toLowerCase().includes(lowerQuery))
        );
        this.renderModelOptions(filtered);
    }

    renderModelOptions(models) {
        this.modelSelect.innerHTML = '';

        // Add a "Refresh List" option at the top
        const refreshOpt = document.createElement('option');
        refreshOpt.value = "_refresh";
        refreshOpt.textContent = "🔄 Refresh Model List";
        this.modelSelect.appendChild(refreshOpt);

        if (Array.isArray(models) && models.length > 0) {
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_name;
                option.textContent = model.title;
                // Store arch in data attribute
                if (model.architecture) {
                    option.dataset.arch = model.architecture;
                }
                this.modelSelect.appendChild(option);
            });
        } else {
            const option = document.createElement('option');
            option.value = "";
            option.textContent = "No matching models found";
            option.disabled = true;
            this.modelSelect.appendChild(option);
        }
    }

    // Load available models
    async loadModels() {
        if (!this.modelSelect) return;

        this.modelSelect.innerHTML = '<option value="" disabled selected>Loading models...</option>';
        this.modelSelect.disabled = true;

        if (this.modelStatus) {
            this.modelStatus.classList.remove('hidden');
        }

        try {
            const endpoint = `${this.apiEndpoint}/sdapi/v1/sd-models`;
            console.log(`Fetching models from: ${endpoint}`);

            const response = await fetch(endpoint).catch(err => {
                throw new Error(`Network error: ${err.message}. Check if your browser or Brave Shield is blocking requests to ${endpoint}`);
            });

            if (!response.ok) {
                const errorText = await response.text().catch(() => '');
                throw new Error(`API returned ${response.status}: ${response.statusText}. ${errorText.substring(0, 100)}`);
            }

            const models = await response.json();
            this.availableModelsList = models;
            console.log('Received models from API:', JSON.stringify(models));

            this.renderModelOptions(models);
            console.log(`Injected ${models.length} models into dropdown`);

            // Show search input if models loaded
            if (this.modelSearchInput) {
                this.modelSearchInput.style.display = 'block';
            }

            // Select current model logic
            this.checkCurrentModel();

        } catch (error) {
            console.error('Error loading models:', error);
            this.modelSelect.innerHTML = `
                <option value="" disabled selected>Error loading models</option>
                <option value="_refresh">🔄 Try Again</option>
            `;
            this.showNotification(`Could not load models: ${error.message}`, 'error');
        } finally {
            this.modelSelect.disabled = false;
            if (this.modelStatus) {
                this.modelStatus.classList.add('hidden');
            }
        }
    }

    // Helper to highlight active model
    updateModelHighlight() {
        if (!this.modelSelect) return;
        Array.from(this.modelSelect.options).forEach(opt => {
            // Highlight the actually loaded model, not just the selected one
            if (opt.value === this.currentlyLoadedModel) {
                opt.style.color = '#ef4444'; // Red
                opt.style.fontWeight = 'bold';
            } else {
                opt.style.color = '';
                opt.style.fontWeight = 'normal';
            }
        });
    }

    // Check currently active model
    async checkCurrentModel() {
        try {
            // Use options endpoint which is proxy-friendly
            const endpoint = `${this.apiEndpoint}/sdapi/v1/options`;
            const response = await fetch(endpoint);

            if (response.ok) {
                const data = await response.json();
                // AUTOMATIC1111 options returns sd_model_checkpoint
                if (data.sd_model_checkpoint) {
                    this.currentlyLoadedModel = data.sd_model_checkpoint; // Track loaded model
                    this.modelSelect.value = data.sd_model_checkpoint;
                    // Update arch based on selected value
                    const selectedOption = this.modelSelect.selectedOptions[0];
                    if (selectedOption && selectedOption.dataset.arch) {
                        this.currentModelArch = selectedOption.dataset.arch;
                    }
                } else if (data.current_model) {
                    this.currentlyLoadedModel = data.current_model; // Track loaded model
                    this.modelSelect.value = data.current_model;
                    const selectedOption = this.modelSelect.selectedOptions[0];
                    if (selectedOption && selectedOption.dataset.arch) {
                        this.currentModelArch = selectedOption.dataset.arch;
                    }
                }
            }

            // Highlight active model
            this.updateModelHighlight();
        } catch (e) {
            console.warn('Could not check current model:', e);
        }
    }

    // Switch model
    async switchModel() {
        const selectedModel = this.modelSelect.value;
        if (!selectedModel) return;

        this.modelSelect.disabled = true;
        if (this.modelStatus) {
            this.modelStatus.classList.remove('hidden');
        }
        this.showNotification('Switching model... This may take a moment.', 'info');

        try {
            const endpoint = `${this.apiEndpoint}/sdapi/v1/options`;
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sd_model_checkpoint: selectedModel })
            });

            if (!response.ok) throw new Error('Failed to switch model');

            // Track the newly loaded model
            this.currentlyLoadedModel = selectedModel;

            // Update local state architecture
            const selectedOption = this.modelSelect.selectedOptions[0];
            if (selectedOption && selectedOption.dataset.arch) {
                this.currentModelArch = selectedOption.dataset.arch;
                console.log(`Current architecture updated to: ${this.currentModelArch}`);
            }

            // Update the visual highlight immediately
            this.updateModelHighlight();

            this.showNotification(`Model switched to ${this.currentModelArch.toUpperCase()} mode!`, 'success');
        } catch (error) {
            console.error('Error switching model:', error);
            this.showNotification(`Failed to switch model: ${error.message}`, 'error');
            // Revert selection if possible or reload
            this.checkCurrentModel();
        } finally {
            this.modelSelect.disabled = false;
            if (this.modelStatus) {
                this.modelStatus.classList.add('hidden');
            }
        }
    }

    // Handle manual model upload
    async handleModelUpload() {
        const file = this.modelUploadInput.files[0];
        if (!file) {
            this.showNotification('Please select a file first', 'warning');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);
        formData.append('filename', file.name);

        this.uploadProgressContainer.classList.remove('hidden');
        this.uploadModelBtn.disabled = true;
        this.browseModelBtn.disabled = true;

        try {
            const endpoint = `${this.apiEndpoint}/sdapi/v1/upload-model`;

            // Use XHR for progress tracking
            const xhr = new XMLHttpRequest();

            // Update progress bar
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    this.uploadProgressBar.style.width = `${percent}%`;
                    this.uploadProgressText.textContent = `${percent}% (${(e.loaded / 1024 / 1024).toFixed(1)} / ${(e.total / 1024 / 1024).toFixed(1)} MB)`;
                }
            });

            // Handle completion
            const uploadPromise = new Promise((resolve, reject) => {
                xhr.onload = () => {
                    if (xhr.status >= 200 && xhr.status < 300) {
                        resolve(JSON.parse(xhr.responseText));
                    } else {
                        reject(new Error(`Upload failed with status ${xhr.status}`));
                    }
                };
                xhr.onerror = () => reject(new Error('Network error during upload'));
            });

            xhr.open('POST', endpoint);
            xhr.send(formData);

            await uploadPromise;

            this.showNotification(`Model ${file.name} uploaded successfully!`, 'success');
            this.modelUploadInput.value = ''; // Clear input
            this.loadModels(); // Refresh model list to show new local model
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.uploadProgressContainer.classList.add('hidden');
            this.uploadModelBtn.disabled = false;
            this.browseModelBtn.disabled = false;
            this.uploadProgressBar.style.width = '0%';
        }
    }

    // Switch between Model Management tabs
    switchTab(tabName) {
        this.tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        this.tabContents.forEach(content => {
            content.classList.toggle('hidden', content.id !== `${tabName}Tab`);
        });
    }

    // Search models on Civitai via backend proxy
    async searchCivitai() {
        const query = this.civitaiSearchInput.value.trim();
        if (!query) {
            this.showNotification('Please enter a search query', 'warning');
            return;
        }

        this.civitaiResults.innerHTML = '<div style="text-align: center; padding: 2rem;"><span class="loading-spinner"></span> Searching...</div>';
        this.civitaiSearchBtn.disabled = true;

        try {
            const response = await fetch(`${this.apiEndpoint}/sdapi/v1/civitai/models?query=${encodeURIComponent(query)}`);
            if (!response.ok) throw new Error('Search failed');

            const data = await response.json();
            this.displayCivitaiResults(data.items);
        } catch (error) {
            console.error('Civitai search error:', error);
            this.showNotification('Failed to search Civitai', 'error');
            this.civitaiResults.innerHTML = '<div style="text-align: center; color: var(--color-error); padding: 1rem;">Search failed. Try again.</div>';
        } finally {
            this.civitaiSearchBtn.disabled = false;
        }
    }

    // Display search results from Civitai
    displayCivitaiResults(items) {
        if (!items || items.length === 0) {
            this.civitaiResults.innerHTML = '<div style="text-align: center; padding: 1rem;">No models found</div>';
            return;
        }

        this.civitaiResults.innerHTML = '';
        items.forEach(item => {
            const el = document.createElement('div');
            el.className = 'civitai-item';
            el.innerHTML = `
                <img src="${item.image || ''}" class="civitai-img" onerror="this.src='https://via.placeholder.com/60?text=No+Img'">
                <div class="civitai-info">
                    <div class="civitai-name" title="${item.name}">${this.escapeHtml(item.name)}</div>
                    <div class="civitai-meta">${item.type || 'Model'} • ${item.filename || 'Unknown file'}</div>
                </div>
                <button class="btn btn-primary btn-sm download-btn" data-url="${item.downloadUrl}" data-name="${item.filename}" data-type="${item.type}" title="Download Model">⬇️</button>
            `;

            const downBtn = el.querySelector('.download-btn');
            downBtn.addEventListener('click', () => this.downloadFromCivitai(item.downloadUrl, item.filename, item.type));

            this.civitaiResults.appendChild(el);
        });
    }

    // Start background download from Civitai via backend proxy
    async downloadFromCivitai(url, filename, type = 'Checkpoint') {
        const isLora = type.toUpperCase() === 'LORA';
        const displayType = isLora ? 'LoRA' : 'Model';
        this.showNotification(`Starting ${displayType} download: ${filename}`, 'info');

        try {
            const formData = new FormData();
            formData.append('downloadUrl', url);
            formData.append('filename', filename);

            // Add a flag for LoRA so the backend knows where to save it
            if (isLora) formData.append('isLora', 'true');

            const response = await fetch(`${this.apiEndpoint}/sdapi/v1/civitai/download`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Download request failed');

            this.showNotification(`${displayType} download started. It will appear in your list once finished.`, 'success');
        } catch (error) {
            console.error('Civitai download error:', error);
            this.showNotification('Failed to start download', 'error');
        }
    }

    // Handle manual URL download
    async handleDirectDownload() {
        const url = this.directDownloadUrl.value.trim();
        const type = this.directDownloadType.value;

        if (!url) {
            this.showNotification('Please enter a download URL', 'warning');
            return;
        }

        if (!url.startsWith('http')) {
            this.showNotification('Invalid URL format', 'error');
            return;
        }

        // Try to extract a friendly name from Civitai URLs
        let filename = 'downloaded_model';
        if (url.includes('civitai.com')) {
            const modelIdMatch = url.match(/models\/(\d+)/);
            if (modelIdMatch) {
                filename = `civitai_${modelIdMatch[1]}`;
            }
        }

        this.directDownloadBtn.disabled = true;
        const originalText = this.directDownloadBtn.textContent;
        this.directDownloadBtn.textContent = '⏳ Starting...';

        try {
            await this.downloadFromCivitai(url, filename, type);
            this.directDownloadUrl.value = '';
        } finally {
            this.directDownloadBtn.disabled = false;
            this.directDownloadBtn.textContent = originalText;
        }
    }

    // Apply quality presets
    applyPreset(type) {
        const isXL = this.currentModelArch === 'sdxl';

        if (type === 'portrait') {
            this.widthInput.value = isXL ? 896 : 512;
            this.heightInput.value = isXL ? 1152 : 768;
            this.stepsInput.value = isXL ? 25 : 30; // XL needs fewer steps usually
            this.cfgInput.value = isXL ? 5.0 : 7.5; // XL often better with lower CFG
        } else {
            this.widthInput.value = isXL ? 1152 : 768;
            this.heightInput.value = isXL ? 896 : 512;
            this.stepsInput.value = isXL ? 25 : 30;
            this.cfgInput.value = isXL ? 5.0 : 7.5;
        }

        // Trigger input events to update UI state if needed
        this.widthInput.dispatchEvent(new Event('change'));
        this.heightInput.dispatchEvent(new Event('change'));

        // Update displays
        this.stepsValue.textContent = this.stepsInput.value;
        this.cfgValue.textContent = this.cfgInput.value;
        this.showNotification(`Applied ${type} preset for ${this.currentModelArch.toUpperCase()}`, 'info');

        if (isXL) {
            this.showNotification('Note: SDXL works best at high resolutions (1024x1024 area).', 'warning');
        }
    }

    // Fetch available LoRAs from backend
    // Load LoRAs from localStorage
    loadLorasFromStorage() {
        try {
            const saved = localStorage.getItem('sd_selected_loras');
            if (saved) {
                this.selectedLoras = JSON.parse(saved);
                this.renderActiveLoras();
            }
        } catch (e) {
            console.error('Error loading saved LoRAs', e);
        }
    }

    // Save LoRAs to localStorage
    saveLorasToStorage() {
        localStorage.setItem('sd_selected_loras', JSON.stringify(this.selectedLoras));
    }

    async fetchLoras() {
        try {
            const response = await fetch(`${this.apiEndpoint}/sdapi/v1/loras`);
            if (!response.ok) throw new Error('Failed to fetch LoRAs');
            this.availableLoras = await response.json();
            console.log(`Fetched ${this.availableLoras.length} LoRAs`);
            this.renderActiveLoras(); // In case some were pre-selected
        } catch (error) {
            console.error('Error fetching LoRAs:', error);
        }
    }

    // LoRA Selection Modal
    openLoraModal() {
        if (!this.loraModal) return;
        this.loraModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
        this.loraSearchInput.value = '';
        this.loraSearchInput.focus();
        this.renderLoraList();
    }

    closeLoraModal() {
        if (!this.loraModal) return;
        this.loraModal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    searchLoras() {
        this.renderLoraList();
    }

    renderLoraList() {
        const query = this.loraSearchInput.value.toLowerCase();
        const filtered = this.availableLoras.filter(lora =>
            lora.name.toLowerCase().includes(query) ||
            lora.title.toLowerCase().includes(query)
        );

        this.loraListDisplay.innerHTML = '';

        if (filtered.length === 0) {
            this.loraListDisplay.innerHTML = '<div class="text-center text-muted" style="padding: 1rem;">No LoRAs found</div>';
            return;
        }

        filtered.forEach(lora => {
            const isSelected = this.selectedLoras.some(s => s.name === lora.name);

            const item = document.createElement('div');
            item.className = `lora-item ${isSelected ? 'selected' : ''}`;
            item.style.cursor = 'pointer';
            item.style.marginBottom = '0.5rem';

            item.innerHTML = `
            <div class="lora-item-row">
                <div style="flex: 1; overflow: hidden; padding-right: 0.5rem;">
                    <div class="lora-item-title">${this.escapeHtml(lora.title)}</div>
                    <div class="lora-item-subtitle">${this.escapeHtml(lora.name)}</div>
                </div>
                <button type="button" class="btn ${isSelected ? 'btn-secondary' : 'btn-primary'} btn-sm" style="flex-shrink: 0;">
                    ${isSelected ? 'Selected' : 'Add'}
                </button>
            </div>
        `;

            item.onclick = (e) => {
                if (isSelected) {
                    this.removeLora(lora.name);
                } else {
                    this.addLora(lora);
                }
                this.renderLoraList();
            };

            this.loraListDisplay.appendChild(item);
        });
    }

    addLora(lora) {
        if (!this.selectedLoras.some(s => s.name === lora.name)) {
            this.selectedLoras.push({ name: lora.name, weight: 1.0, title: lora.title });
            this.saveLorasToStorage();
            this.renderActiveLoras();
            this.showNotification(`Added LoRA: ${lora.title}`, 'success');
        }
    }

    removeLora(loraName) {
        this.selectedLoras = this.selectedLoras.filter(s => s.name !== loraName);
        this.saveLorasToStorage();
        this.renderActiveLoras();
        // Update list if modal is open
        if (!this.loraModal.classList.contains('hidden')) {
            this.renderLoraList();
        }
    }

    updateLoraWeight(loraName, weight) {
        const lora = this.selectedLoras.find(s => s.name === loraName);
        if (lora) {
            lora.weight = parseFloat(weight);
            this.saveLorasToStorage();
        }
    }

    renderActiveLoras() {
        if (!this.activeLorasList) return;
        this.activeLorasList.innerHTML = '';

        if (this.selectedLoras.length === 0) {
            this.activeLorasList.appendChild(this.noLorasMessage);
            return;
        }

        this.selectedLoras.forEach(lora => {
            const item = document.createElement('div');
            item.className = 'lora-item';
            item.style.marginBottom = '0.5rem';

            item.innerHTML = `
            <div class="lora-item-row">
                <div class="lora-item-title" style="flex: 1; max-width: 140px;" title="${this.escapeHtml(lora.title)}">
                    ${this.escapeHtml(lora.title)}
                </div>
                <button type="button" class="btn btn-secondary btn-sm" style="padding: 2px 6px; font-size: 0.7rem;">✕</button>
            </div>
            <div style="display: grid; grid-template-columns: 1fr auto; gap: 0.75rem; align-items: center; margin-top: 0.25rem;">
                <input type="range" class="form-range" min="0" max="1.5" step="0.05" value="${lora.weight}">
                <span style="font-size: 0.75rem; font-family: monospace; min-width: 30px; text-align: right; line-height: 1.5;">${lora.weight.toFixed(2)}</span>
            </div>
        `;

            // Remove button
            item.querySelector('button').onclick = (e) => {
                e.stopPropagation();
                this.removeLora(lora.name);
            };

            // Weight slider
            const slider = item.querySelector('input[type="range"]');
            const display = item.querySelector('span');
            slider.oninput = (e) => {
                const val = parseFloat(e.target.value);
                display.textContent = val.toFixed(2);
                this.updateLoraWeight(lora.name, val);
            };

            this.activeLorasList.appendChild(item);
        });
    }

    // Saved Prompts Logic
    saveCurrentPrompt() {
        const prompt = this.promptInput.value.trim();
        const negative = this.negativePromptInput.value.trim();

        if (!prompt) {
            this.showNotification('Please enter a prompt to save', 'warning');
            return;
        }

        // Simple prompt for name
        const name = window.prompt('Enter a name for this prompt:', prompt.substring(0, 30) + '...');
        if (!name) return; // User cancelled

        const newPrompt = {
            id: Date.now(),
            name: name,
            prompt: prompt,
            refinement_prompt: this.refinementPromptInput.value.trim(),
            negative_prompt: negative,
            timestamp: new Date().toISOString()
        };

        this.savedPrompts.unshift(newPrompt);
        this.savePromptsToStorage();
        this.showNotification('Prompt saved successfully!', 'success');
    }

    savePromptsToStorage() {
        localStorage.setItem('sd_saved_prompts', JSON.stringify(this.savedPrompts));
    }

    openPromptsModal() {
        if (!this.promptsModal) return;
        this.renderPromptsList();
        this.promptsModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }

    closePromptsModal() {
        if (!this.promptsModal) return;
        this.promptsModal.classList.add('hidden');
        document.body.style.overflow = '';
    }

    renderPromptsList() {
        if (!this.promptsList) return;
        this.promptsList.innerHTML = '';

        if (this.savedPrompts.length === 0) {
            this.promptsList.innerHTML = '<div class="text-center text-muted" style="padding: 1rem;">No saved prompts yet.</div>';
            return;
        }

        this.savedPrompts.forEach(p => {
            const item = document.createElement('div');
            item.className = 'saved-prompt-item';

            // Safe helper for HTML escaping
            const escape = (str) => {
                if (!str) return '';
                return str.replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;')
                    .replace(/"/g, '&quot;')
                    .replace(/'/g, '&#039;');
            };

            item.innerHTML = `
                <div class="saved-prompt-text">
                    <div class="saved-prompt-name">${escape(p.name)}</div>
                    <div class="saved-prompt-preview">${escape(p.prompt)}</div>
                </div>
                <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <button class="btn btn-primary btn-sm user-select-none load-btn" style="padding: 2px 8px;">Load</button>
                    <button class="btn btn-secondary btn-sm user-select-none delete-btn" style="padding: 2px 8px; color: #ef4444; border-color: rgba(239, 68, 68, 0.3);">✕</button>
                </div>
            `;

            // Load button
            item.querySelector('.load-btn').onclick = () => {
                this.loadPromptIntoForm(p);
                this.closePromptsModal();
            };

            // Delete button
            item.querySelector('.delete-btn').onclick = (e) => {
                e.stopPropagation();
                if (confirm(`Delete "${p.name}"?`)) {
                    this.deletePrompt(p.id);
                }
            };

            this.promptsList.appendChild(item);
        });
    }

    loadPromptIntoForm(saved) {
        this.promptInput.value = saved.prompt;
        this.refinementPromptInput.value = saved.refinement_prompt || '';
        this.negativePromptInput.value = saved.negative_prompt || '';
        this.updateCharCount();
        this.showNotification(`Loaded "${saved.name}"`, 'success');
    }

    deletePrompt(id) {
        this.savedPrompts = this.savedPrompts.filter(p => p.id !== id);
        this.savePromptsToStorage();
        this.renderPromptsList();
        this.showNotification('Prompt deleted', 'info');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.sdApp = new StableDiffusionApp();
});

// Add slide out animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);
