const path = require('path');
const {PythonShell} = require('python-shell');
const fs = require('fs');

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
    res.render('page/home.pug', { error: null }); // hoặc chỉ hiển thị form
};

// POST - Xử lý dự đoán
module.exports.PostHome = async (req, res) => {
    const userData = req.body;

    // Lấy tên sinh viên (tùy chọn)
    const studentName = userData.student_name?.trim() || 'Bạn';
    delete userData.student_name; // không gửi tên vào model

    try {
        const output = await runPython(userData);

        // Nếu Python trả về lỗi
        if (output.error) {
            return res.render('page/home.pug', {
                error: output.error,
                oldData: req.body
            });
        }

        // Ghi log (điểm cộng đồ án)
        const logLine = `${new Date().toISOString()} | ${studentName} | ${output.probability}%\n`;
        fs.appendFileSync('predictions.log', logLine);

        // Xác định mức độ nguy cơ
        const level = output.probability < 30 ? 'low'
                    : output.probability < 70 ? 'medium'
                    : 'high';

        // Render trang kết quả
        res.render('page/result.pug', {
            name: studentName,
            probability: output.probability,
            level: level
        });

    } catch (err) {
        console.error('Lỗi khi gọi Python:', err);
        res.render('page/home.pug', {
            error: 'Hệ thống gặp sự cố, vui lòng thử lại sau ít phút!',
            oldData: req.body
        });
    }
};