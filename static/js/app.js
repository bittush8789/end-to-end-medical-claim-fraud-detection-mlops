document.addEventListener("DOMContentLoaded", () => {
    // Interactive progress ring if it exists in the page
    const riskRing = document.getElementById("risk-ring-circle");
    if (riskRing) {
        const radius = riskRing.r.baseVal.value;
        const circumference = radius * 2 * Math.PI;
        riskRing.style.strokeDasharray = `${circumference} ${circumference}`;
        
        const probability = parseFloat(riskRing.getAttribute("data-probability")) || 0;
        const offset = circumference - (probability / 100) * circumference;
        
        // Animating stroke offset
        setTimeout(() => {
            riskRing.style.strokeDashoffset = offset;
        }, 300);
    }
});

// Mock scenarios to populate the form instantly
const scenarios = {
    legitimate: {
        patient_age: 45,
        gender: "Female",
        policy_type: "Gold",
        policy_tenure: 36,
        previous_claims: 1,
        claim_amount: 3200,
        hospital_type: "Public",
        procedure_code: "P001",
        diagnosis_code: "D001",
        length_of_stay: 3,
        emergency_admission: 0,
        provider_id: "PROV0032",
        provider_claim_count: 85,
        previous_fraud_cases: 0,
        coverage_amount: 50000,
        deductible_amount: 250
    },
    fraudulent: {
        patient_age: 26,
        gender: "Male",
        policy_type: "Bronze",
        policy_tenure: 8,
        previous_claims: 9,
        claim_amount: 45000,
        hospital_type: "Private",
        procedure_code: "P004",
        diagnosis_code: "D001",
        length_of_stay: 14,
        emergency_admission: 1,
        provider_id: "PROV0080",
        provider_claim_count: 120,
        previous_fraud_cases: 8,
        coverage_amount: 30000,
        deductible_amount: 1000
    }
};

function populateForm(type) {
    const data = scenarios[type];
    if (!data) return;

    // Fill all form inputs
    Object.keys(data).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = data[key];
        }
    });

    // Notify user with a brief glow or visual cue on inputs
    const inputs = document.querySelectorAll(".form-control, .form-select");
    inputs.forEach(el => {
        el.style.borderColor = "#06b6d4";
        setTimeout(() => {
            el.style.borderColor = "";
        }, 1000);
    });
}
