const express = require("express")
const router = express.Router()
const controller = require("../controller/home.controller")

router.get('/', controller.home )
router.post('/', controller.PostHome)

module.exports = router