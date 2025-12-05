const path = require('path');
const {PythonShell} = require('python-shell');
const fs = require('fs');


const MODEL_OPTIONS = [
    'RandomForestClassifier',
    'DecisionTreeClassifier',
    'GradientBoostingClassifier',
    'LogisticRegression',
    'LinearSVC',
    'SVC',
];

const DEFAULT_MODEL = 'RandomForestClassifier';

const runPython = async (userData) => {
    const options = {
        mode: 'text',
        pythonPath: 'C:\\Users\\trita\\AppData\\Local\\Programs\\Python\\Python313\\python.exe',  // Python 3.13 có joblib
        pythonOptions: ['-u'],                    // unbuffered output
        scriptPath: path.join(__dirname, '..'),  // thư mục chứa predict.py (ở root)
        args: [JSON.stringify(userData)]
    };

    try {
        const results = await PythonShell.run('predict.py', options);
        
        if (!results || results.length === 0) {
            throw new Error('Python không trả kết quả');
        }

        try {
            const data = JSON.parse(results[0]);
            return data;
        } catch (e) {
            throw new Error('Kết quả từ Python không phải JSON');
        }
    } catch (err) {
        throw err;
    }
};

// GET - Hiển thị form
module.exports.home = async (req, res) => {
    res.render('page/home.pug', { error: null, models: MODEL_OPTIONS }); // hoặc chỉ hiển thị form
};

// POST - Xử lý dự đoán
module.exports.PostHome = async (req, res) => {
    const userData = req.body;
    const inputSnapshot = { ...req.body };

    const studentName = userData.student_name?.trim() || 'Bạn';
    delete userData.student_name; // không gửi tên vào model

    try {
        const output = await runPython(userData);

        // Nếu Python trả về lỗi
        if (output.error) {
            return res.render('page/home.pug', {
                error: output.error,
                oldData: req.body,
                models: MODEL_OPTIONS,
            });
        }

        const selectedModel = output.model_name || userData.model_name || DEFAULT_MODEL;
        const probability = output.probability;
        const comparison = output.probabilities || {};

        // Ghi log (điểm cộng đồ án)
        const sanitizedInputs = JSON.stringify(inputSnapshot).replace(/\s+/g, ' ');
        const logLine = `${new Date().toISOString()} | ${studentName} | ${selectedModel} | ${probability}% | ${sanitizedInputs}\n`;
        fs.appendFileSync('predictions.csv', logLine);

        // Xác định mức độ nguy cơ
        const level = probability < 30 ? 'low'
                    : probability < 70 ? 'medium'
                    : 'high';

        // Render trang kết quả
        res.render('page/result.pug', {
            name: studentName,
            probability: probability,
            level: level,
            selectedModel,
            probabilities: comparison,
            inputs: inputSnapshot,
        });

    } catch (err) {
        console.error('Lỗi khi gọi Python:', err);
        res.render('page/home.pug', {
            error: 'Hệ thống gặp sự cố, vui lòng thử lại sau ít phút!',
            oldData: req.body,
            models: MODEL_OPTIONS,
        });
    }
};

// GET - Hiển thị form dự đoán theo student_id
module.exports.predictByStudent = async (req, res) => {
    res.render('page/predict-student.pug', { 
        error: null, 
        models: MODEL_OPTIONS,
        persisted: req.query || {}
    });
};

// POST - Xử lý dự đoán theo student_id
module.exports.PostPredictByStudent = async (req, res) => {
    const { student_id, weekday, weather, model_name } = req.body;

    // Validation
    if (!student_id || !weekday || !weather) {
        return res.render('page/predict-student.pug', {
            error: 'Vui lòng điền đầy đủ thông tin: Mã sinh viên, Thứ ngày và Thời tiết',
            models: MODEL_OPTIONS,
            persisted: req.body,
        });
    }

    try {
        const userData = {
            student_id: student_id.trim(),
            weekday: weekday,
            weather: weather,
            model_name: model_name || DEFAULT_MODEL
        };

        const output = await runPython(userData);

        // Nếu Python trả về lỗi
        if (output.error) {
            return res.render('page/predict-student.pug', {
                error: output.error,
                models: MODEL_OPTIONS,
                persisted: req.body,
            });
        }

        const selectedModel = output.model_name || model_name || DEFAULT_MODEL;
        const probability = output.probability;
        const comparison = output.probabilities || {};
        const studentInfo = output.student_info || {};
        const inputData = output.input_data || {};

        // Ghi log
        const logLine = `${new Date().toISOString()} | ${student_id} | ${selectedModel} | ${probability}% | weekday:${weekday}, weather:${weather}\n`;
        fs.appendFileSync('predictions.csv', logLine);

        // Xác định mức độ nguy cơ
        const level = probability < 30 ? 'low'
                    : probability < 70 ? 'medium'
                    : 'high';

        // Render trang kết quả
        res.render('page/result-student.pug', {
            studentId: student_id,
            weekday: weekday,
            weather: weather,
            probability: probability,
            level: level,
            selectedModel,
            probabilities: comparison,
            studentInfo: studentInfo,
            inputData: inputData,
        });

    } catch (err) {
        console.error('Lỗi khi gọi Python:', err);
        res.render('page/predict-student.pug', {
            error: err.message || 'Hệ thống gặp sự cố, vui lòng thử lại sau ít phút!',
            models: MODEL_OPTIONS,
            persisted: req.body,
        });
    }
};