var modal = document.getElementById("myModal");
var btn = document.getElementById("login");
var submit = document.getElementById("form").submit;
var span = document.getElementsByClassName("close")[0];

var uname = document.getElementById("uname");
var pass = document.getElementById("pass")

let username = "admin"
let password = "password"
 
btn.onclick = function() {
    modal.style.display = "block";
}
span.onclick = function() {
    modal.style.display = "none";
}
window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}
submit.onclick = function() {
    console.log("worked")
    if (uname.value = username && pass.value == password) {
        alert("You have successfully logged in.");
        modal.style.display = "none";
    } else {
        alert("Incorrect login. Please try again.");
    }
}