const video = document.getElementById('webcam-video');
const fileInput = document.getElementById('file-input');
const actionBtn = document.getElementById('action-btn');
const resultImg = document.getElementById('result-img');
const predictionsList = document.getElementById('predictions-list');
const loadingSpinner = document.getElementById('loading-spinner');
const resultPlaceholder = document.getElementById('result-placeholder');
const uploadPlaceholder = document.getElementById('upload-placeholder');

// Stat elements
const statLatency = document.getElementById('stat-latency');
const statFaces = document.getElementById('stat-faces');

// Identify the container that will act as the Drag and Drop zone
const dropZone = uploadPlaceholder.parentElement;

let stream = null;
let isLive = false;

// API Configuration - works for both localhost and production
const getApiUrl = () => {
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        return 'http://localhost:8001';
    } else {
        return `${window.location.protocol}//${window.location.host}`;
    }
};
const API_URL = getApiUrl();

// 1. Source: Upload
document.getElementById('mode-upload').onclick = () => {
    isLive = false;
    stopWebcam();
    fileInput.click();
};

// --- DRAG AND DROP FUNCTIONALITY ---
['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.add('border-blue-500', 'bg-blue-500/10', 'ring-2', 'ring-blue-500/20');
    }, false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault(); e.stopPropagation();
        dropZone.classList.remove('border-blue-500', 'bg-blue-500/10', 'ring-2', 'ring-blue-500/20');
    }, false);
});

dropZone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) uploadAndPredict(file);
}, false);

uploadPlaceholder.onclick = () => {
    if (!video.classList.contains('hidden')) return;
    fileInput.click();
};

// 2. Source: Webcam
document.getElementById('mode-webcam').onclick = async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.classList.remove('hidden');
        uploadPlaceholder.classList.add('hidden');
        actionBtn.classList.remove('hidden');
        actionBtn.innerText = "START LIVE ANALYSIS";
    } catch (err) {
        alert("Camera not found.");
    }
};

function stopWebcam() {
    if (stream) stream.getTracks().forEach(t => t.stop());
    video.classList.add('hidden');
    uploadPlaceholder.classList.remove('hidden');
}

// 3. Inference Logic
actionBtn.onclick = () => {
    isLive = !isLive;
    if (isLive) {
        actionBtn.innerText = "STOP ANALYSIS";
        actionBtn.classList.replace('bg-blue-600', 'bg-red-600');
        runInferenceLoop();
    } else {
        actionBtn.innerText = "START LIVE ANALYSIS";
        actionBtn.classList.replace('bg-red-600', 'bg-blue-600');
    }
};

async function runInferenceLoop() {
    if (!isLive) return;

    const canvas = document.getElementById('capture-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    
    canvas.toBlob(async (blob) => {
        const startTime = performance.now();
        const formData = new FormData();
        formData.append('file', blob);

        try {
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            const endTime = performance.now();
            
            if (result.success) {
                // Update UI Stats
                statLatency.innerText = `${Math.round(endTime - startTime)} ms`;
                statFaces.innerText = result.data.length;

                resultImg.src = result.image;
                resultImg.classList.remove('hidden');
                resultPlaceholder.classList.add('hidden');
                
                predictionsList.innerHTML = result.data.map(p => `
                    <div class="bg-slate-800 p-3 rounded-lg border border-slate-700">
                        <p class="text-emerald-400 font-bold">${p.emotion}</p>
                        <p class="text-white text-xs">Age: ~${Math.round(p.age)}</p>
                    </div>
                `).join('');
            }
        } catch (e) { console.error(e); }
        
        if (isLive) setTimeout(runInferenceLoop, 100);
    }, 'image/jpeg', 0.6);
}

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) uploadAndPredict(file);
};

async function uploadAndPredict(file) {
    const startTime = performance.now();
    loadingSpinner.classList.remove('hidden');
    const formData = new FormData();
    formData.append('file', file);
    try {
        const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData });
        const result = await response.json();
        const endTime = performance.now();

        if (result.success) {
            // Update UI Stats
            statLatency.innerText = `${Math.round(endTime - startTime)} ms`;
            statFaces.innerText = result.data.length;

            resultImg.src = result.image;
            resultImg.classList.remove('hidden');
            resultPlaceholder.classList.add('hidden');
            
            predictionsList.innerHTML = result.data.map(p => `
                <div class="bg-slate-800 p-3 rounded-lg border border-slate-700">
                    <p class="text-emerald-400 font-bold">${p.emotion}</p>
                    <p class="text-white text-xs">Age: ~${Math.round(p.age)}</p>
                </div>
            `).join('');
        }
    } catch(e) { console.error(e); }
    loadingSpinner.classList.add('hidden');
}