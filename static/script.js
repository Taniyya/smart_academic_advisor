document.addEventListener('DOMContentLoaded', function() {
    const studentSelect = document.getElementById('student-select');
    if (studentSelect) {
        studentSelect.addEventListener('change', loadStudentData);
        loadStudentData(); // Load initial data
    }


});

function loadStudentData() {
    const studentId = document.getElementById('student-select').value;
    localStorage.setItem('selected_student', studentId);
    fetch(`/get_student_data/${studentId}`)
        .then(response => response.json())
        .then(data => {
            document.getElementById('student-id').textContent = data.id;
            document.getElementById('student-gender').textContent = data.gender;
            document.getElementById('student-attendance').textContent = data.attendance;
            document.getElementById('student-internal').textContent = data.internal_marks;

            // Calculate performance metrics
            const semesters = [data.sem1, data.sem2, data.sem3, data.sem4, data.sem5, data.sem6];
            const avg = semesters.reduce((a, b) => a + b, 0) / semesters.length;
            document.getElementById('avg-sgpa').textContent = avg.toFixed(2);

            // Best Semester calculation
            const bestSemIndex = semesters.indexOf(Math.max(...semesters));
            document.getElementById('best-sem').textContent = `Sem ${bestSemIndex + 1} (${semesters[bestSemIndex].toFixed(2)})`;

            // Category
            let category = 'Needs Improvement';
            if (avg >= 8) category = 'Excellent';
            else if (avg >= 6) category = 'Good';
            document.getElementById('category').textContent = category;

            // Update chart
            updateChart(semesters);

            // Bind Q&A button
            const qaBtn = document.getElementById('dynamic-qa-btn');
            qaBtn.href = `/qa/${studentId}`;
            qaBtn.style.display = 'inline-flex';
        });
}

let myChart;
function updateChart(semesters) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    if (myChart) {
        myChart.destroy();
    }
    myChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Sem1', 'Sem2', 'Sem3', 'Sem4', 'Sem5', 'Sem6'],
            datasets: [{
                label: 'SGPA',
                data: semesters,
                backgroundColor: 'rgba(37, 99, 235, 0.85)',
                borderColor: 'rgba(37, 99, 235, 1)',
                borderWidth: 1,
                borderRadius: 2
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { color: '#4a4a4a', font: { family: "'Lora', serif" } },
                    grid: { color: 'rgba(0, 0, 0, 0.05)' }
                },
                x: {
                    ticks: { color: '#4a4a4a', font: { family: "'Lora', serif" } },
                    grid: { color: 'rgba(0, 0, 0, 0)' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#0b1528', font: { family: "'Playfair Display', serif", weight: 'bold' } }
                }
            }
        }
    });
}