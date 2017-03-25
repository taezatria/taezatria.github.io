<?php
	#session_start();
	if(isset($_POST['submit'])) {
		$_SESSION['username'] = $_POST['submit'];
		header('location:2.php');
	}
	if(isset($_GET['logout'])){
		session_unset();
		session_destroy();
		echo 'Logout';
	}
?>
<!DOCTYPE html>
<html>
<head>
	<title>
		coba
	</title>
</head>
<body>
<?php
	echo 'HELLO';
?>
	<form action="2.php" method="post" enctype="multipart/form-data">
		<input type="file" name="uploadFile">
		<input type="submit" name="submit" value="Login">
	</form>
</body>
</html>