const form = document.getElementById('form');
const username_input = document.getElementById('firstname-input');
const password_input = document.getElementById('password-input');
const repeat_password_input = document.getElementById('repeat-password-input');
const error_message = document.getElementById('error-message');

form.addEventListener('submit', (e) => {
  let errors = [];

  if (username_input.value.trim() === '') {
    errors.push('Username is required');
    username_input.parentElement.classList.add('incorrect');
  }
  if (password_input.value === '') {
    errors.push('Password is required');
    password_input.parentElement.classList.add('incorrect');
  }
  if (password_input.value.length > 0 && password_input.value.length < 8) {
    errors.push('Password must have at least 8 characters');
    password_input.parentElement.classList.add('incorrect');
  }

  // Only check password match on signup page
  if (repeat_password_input) {
    if (password_input.value !== repeat_password_input.value) {
      errors.push('Passwords do not match');
      password_input.parentElement.classList.add('incorrect');
      repeat_password_input.parentElement.classList.add('incorrect');
    }
  }

  if (errors.length > 0) {
    e.preventDefault();
    error_message.innerText = errors.join('. ');
  }
});

// Clear errors on input
const allInputs = [username_input, password_input, repeat_password_input].filter(input => input != null);
allInputs.forEach(input => {
  input.addEventListener('input', () => {
    if (input.parentElement.classList.contains('incorrect')) {
      input.parentElement.classList.remove('incorrect');
      error_message.innerText = '';
    }
  });
});