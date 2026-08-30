document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const uploadPrompt = document.getElementById("uploadPrompt");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const scanningOverlay = document.getElementById("scanningOverlay");
    const classifyBtn = document.getElementById("classifyBtn");
    const clearBtn = document.getElementById("clearBtn");
    const samplesGrid = document.getElementById("samplesGrid");

    const emptyResultState = document.getElementById("emptyResultState");
    const resultCard = document.getElementById("resultCard");
    const apiStatusBadge = document.getElementById("apiStatusBadge");
    const apiStatusText = document.getElementById("apiStatusText");

    let currentFile = null;
    let allSamples = [];

    // Health check
    async function checkHealth() {
        try {
            const res = await fetch("/health");
            const data = await res.json();
            if (data.status === "healthy" && apiStatusText && apiStatusBadge) {
                apiStatusText.textContent = `Model Ready (${data.device.toUpperCase()})`;
                apiStatusBadge.className = "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold bg-emerald-50 text-emerald-700 border border-emerald-200";
            }
        } catch (e) {
            if (apiStatusText && apiStatusBadge) {
                apiStatusText.textContent = "Offline";
                apiStatusBadge.className = "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold bg-rose-50 text-rose-700 border border-rose-200";
            }
        }
    }

    // Load sample images
    async function loadSamples() {
        try {
            const res = await fetch("/samples");
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                allSamples = data.samples;
                renderSamples();
            } else {
                samplesGrid.innerHTML = `<p class="col-span-3 text-xs text-slate-400 text-center py-3">No samples found.</p>`;
            }
        } catch (e) {
            console.error("Failed to load samples:", e);
        }
    }

    function renderSamples() {
        samplesGrid.innerHTML = "";
        allSamples.forEach(sample => {
            const card = document.createElement("div");
            card.className = "sample-card bg-white hover:bg-emerald-50/40 border border-slate-200 rounded-xl p-2 cursor-pointer text-center transition-all hover:shadow-sm";
            
            const isHealthy = sample.class_name.toLowerCase() === "healthy";
            const label = sample.sub_category || sample.class_name;

            card.innerHTML = `
                <img src="${sample.url}" alt="${label}" class="w-full h-16 object-cover rounded-lg mb-1.5 bg-slate-50 border border-slate-100">
                <p class="text-[10px] font-bold text-slate-800 truncate">${label}</p>
                <span class="inline-flex items-center gap-0.5 mt-1 text-[9px] font-extrabold px-2 py-0.5 rounded-full ${isHealthy ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-rose-50 text-rose-700 border border-rose-200'}">
                    ${isHealthy ? '<i class="fa-solid fa-check text-[8px]"></i> Healthy' : '<i class="fa-solid fa-triangle-exclamation text-[8px]"></i> Diseased'}
                </span>
            `;

            card.addEventListener("click", async () => {
                try {
                    const imgRes = await fetch(sample.url);
                    const blob = await imgRes.blob();
                    const file = new File([blob], sample.filename, { type: "image/jpeg" });
                    handleFileSelected(file);
                    setTimeout(() => classifyImage(), 150);
                } catch (err) {
                    console.error("Error loading sample:", err);
                }
            });

            samplesGrid.appendChild(card);
        });
    }

    function handleFileSelected(file) {
        if (!file || !file.type.startsWith("image/")) {
            alert("Please select a valid image file (JPG, PNG, WEBP).");
            return;
        }

        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add("hidden");
            previewContainer.classList.remove("hidden");
            clearBtn.classList.remove("hidden");
            classifyBtn.removeAttribute("disabled");
        };
        reader.readAsDataURL(file);
    }

    dropZone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelected(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("border-emerald-500", "bg-emerald-50/30");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("border-emerald-500", "bg-emerald-50/30");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-emerald-500", "bg-emerald-50/30");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    clearBtn.addEventListener("click", () => {
        currentFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        uploadPrompt.classList.remove("hidden");
        previewContainer.classList.add("hidden");
        clearBtn.classList.add("hidden");
        classifyBtn.setAttribute("disabled", "true");
        resultCard.classList.add("hidden");
        emptyResultState.classList.remove("hidden");
    });

    async function classifyImage() {
        if (!currentFile) return;

        scanningOverlay.classList.remove("hidden");
        classifyBtn.setAttribute("disabled", "true");

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const res = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Diagnosis failed.");
            }

            const data = await res.json();
            renderResults(data);

        } catch (error) {
            alert(`Inference failed: ${error.message}`);
        } finally {
            scanningOverlay.classList.add("hidden");
            classifyBtn.removeAttribute("disabled");
        }
    }

    classifyBtn.addEventListener("click", classifyImage);

    function renderResults(data) {
        emptyResultState.classList.add("hidden");
        resultCard.classList.remove("hidden");

        const isHealthy = data.prediction.toLowerCase() === "healthy";

        // Category Tag & Title
        document.getElementById("cropCategoryTag").textContent = isHealthy ? "Healthy Foliage" : "Diseased Foliage";
        document.getElementById("diagnosisTitle").textContent = isHealthy ? "Healthy Plant Leaf" : "Diseased Plant Leaf";
        document.getElementById("pathogenType").textContent = `Pathogen: ${data.pathogen_type}`;

        // Health Badge
        const healthBadge = document.getElementById("healthBadge");
        if (isHealthy) {
            healthBadge.innerHTML = `<i class="fa-solid fa-circle-check text-emerald-600 mr-1"></i> HEALTHY`;
            healthBadge.className = "inline-flex items-center px-4 py-1.5 rounded-full text-xs font-extrabold uppercase tracking-wider bg-emerald-50 text-emerald-700 border border-emerald-200";
        } else {
            healthBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-rose-600 mr-1"></i> DISEASED`;
            healthBadge.className = "inline-flex items-center px-4 py-1.5 rounded-full text-xs font-extrabold uppercase tracking-wider bg-rose-50 text-rose-700 border border-rose-200";
        }

        // Confidence
        document.getElementById("confidenceValue").textContent = data.confidence_percentage;
        document.getElementById("inferenceLatency").textContent = `Latency: ${data.inference_time_ms} ms`;

        // Probabilities Breakdown
        const topProbabilities = document.getElementById("topProbabilities");
        topProbabilities.innerHTML = "";
        
        Object.entries(data.class_probabilities).forEach(([clsName, probVal]) => {
            const pct = (probVal * 100).toFixed(1);
            const isClsHealthy = clsName.toLowerCase() === "healthy";
            const row = document.createElement("div");
            row.className = "flex items-center text-xs justify-between gap-3";
            row.innerHTML = `
                <span class="text-slate-700 font-bold text-xs w-28">${clsName}</span>
                <div class="flex-1 bg-slate-200/80 rounded-full h-2.5 overflow-hidden">
                    <div class="h-full rounded-full transition-all duration-500 ${isClsHealthy ? 'bg-emerald-500' : 'bg-rose-500'}" style="width: ${pct}%"></div>
                </div>
                <span class="text-slate-700 font-mono text-xs w-14 text-right font-extrabold">${pct}%</span>
            `;
            topProbabilities.appendChild(row);
        });

        // Symptoms
        const symptomsList = document.getElementById("symptomsList");
        symptomsList.innerHTML = "";
        data.symptoms.forEach(sym => {
            const li = document.createElement("li");
            li.textContent = sym;
            symptomsList.appendChild(li);
        });

        // Causes
        document.getElementById("causesText").textContent = data.causes;

        // Prevention
        const preventionList = document.getElementById("preventionList");
        preventionList.innerHTML = "";
        data.prevention.forEach(prev => {
            const li = document.createElement("li");
            li.textContent = prev;
            preventionList.appendChild(li);
        });

        // Treatments
        document.getElementById("organicTreatmentText").textContent = data.organic_treatment;
        document.getElementById("chemicalTreatmentText").textContent = data.chemical_treatment;

        if (window.innerWidth < 1024) {
            resultCard.scrollIntoView({ behavior: "smooth" });
        }
    }

    checkHealth();
    //loadSamples();
});
