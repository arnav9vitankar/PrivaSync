import Navbar from "@/frontend/src/components/Navbar";
import Sidebar from "@/frontend/src/components/Sidebar";


export default function DashboardLayout({
children       // Whatever is put inside the DashboardLayout tag will come here
}:{
children?:React.ReactNode   // The component expects a prop called children, and it can be any valid React content
}){

    // flex creates a horizontal flex layout , flex-1 means:"Take all remaining space."
return(

<div className="flex">      


<Sidebar/>


<div className="flex-1">

<Navbar/>


<main className="p-6">

{children}

</main>


</div>


</div>


)

}