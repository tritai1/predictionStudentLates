const express = require("express")
const python_shell = require("python-shell")
const path = require("path")
const router = require("./router/app.route")
app = express()
PORT = 8080
// Middleware để parse form data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Static files (CSS, images, etc.)
app.use(express.static(path.join(__dirname, 'public')));

app.set('views', `${__dirname}/views`);
app.set('view engine', 'pug');

router(app)


app.listen(PORT, ()=>{
    console.log("success", PORT)
})
