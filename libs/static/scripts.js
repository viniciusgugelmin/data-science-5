async function getDataset() {
    const viewEl = document.getElementById('view');

    setLoading(true);

    const callApi = async () => {
        return new Promise((resolve, reject) => {
            fetch('/api/dataset')
                .then((response) => {
                    if (response.ok) {
                        resolve(response.json());
                    } else {
                        reject(new Error('Error retrieving dataset'));
                    }
                })
        });
    }

    const renderDataset = async () => {
        const dataset = await callApi();
        const datasetElement = document.getElementById('dataset');
        datasetElement.innerHTML = JSON.stringify(dataset, null, 2);
    }
    
    const dataset = await callApi();


    setLoading(false);
}

function setLoading(value) {
    const viewEl = document.getElementById('view');
    const buttonsEl = document.querySelectorAll('[data-action="caller"]');

    if (value) {
        viewEl.innerText = 'Carregando...';
    } else if (!value && viewEl.innerText === 'Carregando...') {
        viewEl.innerText = '';
    }

    buttonsEl.forEach((button) => {
        button.disabled = value;
    });


}