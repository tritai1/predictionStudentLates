const express = require("express")
const router = express.Router()
const controller = require("../controller/home.controller")

router.get('/', controller.home )
router.post('/', controller.PostHome)
router.get('/predict-student', controller.predictByStudent)
router.post('/predict-student', controller.PostPredictByStudent)

module.exports = router